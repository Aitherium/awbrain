"""awbrain.engine — the Brain loop: harvest, link, ask, verify, watch.

    harvest  notes/sessions -> a wiki of linked markdown + a claims ledger
             (every claim pinned to the file and line span that supports it)
    ask      embed the question, retrieve ONLY the top-k pages, answer with
             citations — the full history is never loaded into the prompt
    verify   walk a claim back to its evidence span in the source file
    watch    re-harvest changed sources so the brain stays current

The wiki is plain markdown on disk: readable, git-able, and yours — the
opposite of a proprietary memory blob.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Iterable

from .claims import extract_claims
from .embed import Embedder, cosine, resolve_ca
from .store import BrainStore, Chunk, Claim, slugify, stable_id

_SOURCE_EXTS = (".md", ".txt", ".markdown", ".jsonl")
_DEFAULT_BRAIN = Path("brain")
_MAX_CHUNK_CHARS = 700
_MIN_CHUNK_CHARS = 200


def _iter_sources(source_dir: Path) -> Iterable[tuple[Path, str]]:
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SOURCE_EXTS:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if text.strip():
            yield path, text


def _chunk_markdown(path: Path, text: str) -> list[tuple[str, str, int, int]]:
    """Split markdown on headings into (heading, body, start_line, end_line)."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str], int]] = []
    cur_heading = ""
    cur_lines: list[str] = []
    cur_start = 1
    for idx, line in enumerate(lines, start=1):
        if line.startswith("##"):
            if cur_lines:
                sections.append((cur_heading, cur_lines, cur_start))
            cur_heading = line.lstrip("#").strip()
            cur_lines = [line]
            cur_start = idx
        else:
            cur_lines.append(line)
    if cur_lines:
        sections.append((cur_heading, cur_lines, cur_start))

    out: list[tuple[str, str, int, int]] = []
    for heading, sec_lines, start in sections:
        body = "\n".join(sec_lines).strip()
        if not body:
            continue
        # Sub-chunk over-long sections at paragraph boundaries.
        # 🚨 para_start must advance by the CONSUMED lines, not by 1: the old
        # `para_start += len(para_lines) + 1` (after resetting para_lines)
        # shifted every later chunk's start line down by (para_size - 1) per
        # boundary — claims in the second paragraph of a long section were
        # pinned to the WRONG source line (measured 2026-08-26: a claim whose
        # real line held a figure traced to an empty line 82 instead).
        if len(body) > _MAX_CHUNK_CHARS:
            para_lines: list[str] = []
            para_start = start
            for ln in sec_lines:
                para_lines.append(ln)
                if ln.strip() == "" and len("\n".join(para_lines)) >= _MIN_CHUNK_CHARS:
                    consumed = len(para_lines)
                    part = "\n".join(para_lines).strip()
                    out.append((heading, part, para_start, para_start + consumed - 1))
                    para_start += consumed
                    para_lines = []
            if para_lines:
                part = "\n".join(para_lines).strip()
                out.append((heading, part, para_start, para_start + len(para_lines) - 1))
        else:
            out.append((heading, body, start, start + len(sec_lines) - 1))
    return out


def harvest(source_dir: Path, brain_dir: Path | None = None) -> dict:
    """Build or refresh the brain wiki from a folder of notes/sessions."""
    brain = BrainStore(brain_dir or _DEFAULT_BRAIN)
    brain.ensure()
    embedder = Embedder()

    chunks: list[Chunk] = []
    claims: list[Claim] = []
    for path, text in _iter_sources(source_dir):
        rel = str(path.relative_to(source_dir)).replace("\\", "/")
        for heading, body, start, end in _chunk_markdown(path, text):
            slug = slugify(f"{path.stem}-{heading or start}")
            chunks.append(Chunk(
                id=stable_id(rel, str(start), body[:80]),
                slug=slug,
                text=body,
                source=rel,
                start=start,
                end=end,
                heading=heading,
            ))
            claims.extend(extract_claims(body, rel, start, slug))

    if not chunks:
        raise SystemExit(f"no markdown sources found under {source_dir}")

    # Embed (one batch; the router path is batched, lexical is cheap).
    embs = embedder.embed([c.text for c in chunks])
    for chunk, emb in zip(chunks, embs):
        chunk.emb = emb

    # Link pass: each page's top-3 nearest neighbours become [[wikilinks]].
    for i, chunk in enumerate(chunks):
        scored = sorted(
            (cosine(chunk.emb, other.emb), j)
            for j, other in enumerate(chunks) if j != i
        )
        chunk.related = [chunks[j].slug for _, j in scored[-3:]][::-1]

    for chunk in chunks:
        brain.save_page(chunk)
    brain.save_index(chunks)
    brain.save_claims(claims)

    total_bytes = sum(len(c.text) for c in chunks)
    manifest = {
        "source_root": str(source_dir.resolve()),
        "sources": len({c.source for c in chunks}),
        "pages": len(chunks),
        "claims": len(claims),
        "embedder": embedder.mode,
        "index_entries": len(chunks),
        "context_bytes": 0,
        "total_bytes": total_bytes,
        "harvested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    brain.save_manifest(manifest)
    return manifest


def ask(
    question: str,
    brain_dir: Path | None = None,
    k: int = 4,
    chat_url: str | None = None,
    max_tokens: int = 600,
) -> dict:
    """Answer from ONLY the top-k pages — the full history is never loaded.

    The candidate pool is the main brain PLUS every connected provider's
    index: ask works over connected sources, exactly like a data-terminal
    query — without the terminal, the logins, or the vendor cloud."""
    brain_dir = Path(brain_dir or _DEFAULT_BRAIN)
    brain = BrainStore(brain_dir)
    chunks = brain.load_index()
    for label, store in _claim_stores(brain_dir):
        if label == "main":
            continue
        chunks.extend(store.load_index())
    if not chunks:
        raise SystemExit("brain is empty — run `awbrain harvest <dir>` or "
                         "`awbrain connect <provider> --from <dir>` first")
    embedder = Embedder()
    q_emb = embedder.embed_one(question)
    ranked = sorted(
        ((cosine(q_emb, c.emb), c) for c in chunks if c.emb),
        key=lambda pair: pair[0],
        reverse=True,
    )[:k]

    # 2-hop expansion: the wiki's [[wikilinks]] are the graph — pages linked
    # FROM the retrieved set are the "connected subjects" Perplexity markets,
    # and they cost one dict lookup each (no extra embeddings).
    by_slug = {c.slug: c for c in chunks}
    pages: list[Chunk] = [c for _, c in ranked]
    seen = {c.slug for c in pages}
    for _, c in ranked:
        for slug in c.related:
            if slug in seen or slug not in by_slug or len(pages) >= k + 4:
                continue
            pages.append(by_slug[slug])
            seen.add(slug)

    context = "\n\n".join(
        f"[[{c.slug}]] (from {c.source}):\n{c.text}" for c in pages
    )
    sources = [
        {"page": c.slug, "source": c.source, "score": round(score, 3)}
        for score, c in ranked
    ] + [
        {"page": c.slug, "source": c.source, "score": None, "hop": 2}
        for c in pages[k:]
    ]

    answer = _answer_with_router(
        question, context, chat_url=chat_url, max_tokens=max_tokens
    )
    total_bytes = sum(
        store.load_manifest().get("total_bytes", 0)
        for _, store in _claim_stores(brain_dir)
    )
    return {
        "answer": answer,
        "sources": sources,
        "context_bytes": len(context),
        "total_bytes": total_bytes,
        "k": k,
        "pages_retrieved": len(pages),
    }


def _answer_with_router(
    question: str, context: str, chat_url: str | None, max_tokens: int
) -> str | None:
    """Ask the fleet orchestrator to answer from the retrieved pages only."""
    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError:
        return None
    url = chat_url or os.environ.get(
        "AWBRAIN_CHAT_URL", "https://127.0.0.1:8150/v1/chat/completions"
    )
    payload = {
        # "auto" is the router's own default resolution — a neutral id, so the
        # shipped package never advertises our serving roster (AWB005).
        "model": os.environ.get("AWBRAIN_CHAT_MODEL", "auto"),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You answer from the provided wiki pages ONLY. Every claim "
                    "you make must cite the page it came from as [[slug]]. If "
                    "the pages do not answer the question, say so plainly."
                ),
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}\n\nWIKI PAGES:\n{context}",
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    try:
        # The router speaks the fleet's internal CA — verify against it when
        # present, never blindly. On a box without it, the call fails and the
        # honest lexical fallback path reports it.
        resp = httpx.post(
            url, json=payload, timeout=120.0,
            verify=resolve_ca() or True,
        )
        if resp.status_code != 200:
            return None
        return (resp.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


def _claim_stores(brain_dir: Path):
    """(label, BrainStore) for the main brain and every connector sub-brain."""
    yield "main", BrainStore(brain_dir)
    conn_root = brain_dir / "connectors"
    if conn_root.is_dir():
        for entry in sorted(conn_root.iterdir()):
            if entry.is_dir() and (entry / "claims.yaml").is_file():
                yield entry.name, BrainStore(entry)


def verify(claim_text: str, brain_dir: Path | None = None) -> dict:
    """Walk a claim back to the source file span that supports it.

    Searches the main ledger AND every connected provider's ledger — a
    figure from a licensed source is a claim like any other, pinned to the
    source record it came from."""
    brain_dir = Path(brain_dir or _DEFAULT_BRAIN)
    needle = claim_text.lower()
    for label, store in _claim_stores(brain_dir):
        for claim in store.load_claims():
            text = str(claim.get("text", "")).lower()
            if not text:
                continue
            if needle in text or text in needle:
                evidence = claim.get("evidence") or {}
                span = str(evidence.get("span", "0-0"))
                excerpt = _read_excerpt(store, evidence, span)
                return {
                    "found": True,
                    "claim": claim.get("text"),
                    "evidence": {
                        "file": evidence.get("file"),
                        "span": span,
                        "excerpt": excerpt,
                    },
                    "wiki": claim.get("wiki"),
                    "connector": None if label == "main" else label,
                }
    return {"found": False, "claim": claim_text}


def _read_excerpt(store: BrainStore, evidence: dict, span: str) -> str:
    try:
        start_s, _, end_s = span.partition("-")
        start, end = int(start_s), int(end_s or start_s)
        rel = str(evidence.get("file", ""))
        src_root = store.load_manifest().get("source_root", "")
        candidates = []
        if src_root:
            candidates.append(Path(src_root) / rel)
        candidates.append(Path(rel))
        for src in candidates:
            if src.is_file():
                lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
                return "\n".join(lines[max(0, start - 1):end])
    except (OSError, ValueError):
        # The source file may have moved since harvest (e.g. the connector's
        # export dir was archived). Fall back to the stored excerpt — the
        # claim and its span remain in the ledger either way.
        return str(evidence.get("excerpt", "")) or ""
    return str(evidence.get("excerpt", "")) or ""


def connect(
    provider: str,
    source_dir: Path,
    brain_dir: Path | None = None,
    license_ref: str | None = None,
) -> dict:
    """Connect a data provider: ingest its source into the brain and register it.

    The competitor's 'Connectors' page is a list of vendors you may log into;
    this is the same shape without the lock-in — ANY source (a licensed
    portal's export, a mirrored docs site, a HAR-mapped API, local files)
    becomes a connector whose figures trace to the source records via the
    claims ledger.
    """
    brain = BrainStore(brain_dir or _DEFAULT_BRAIN)
    brain.ensure()
    provider_slug = slugify(provider)
    conn_dir = brain_dir or _DEFAULT_BRAIN
    conn_root = Path(conn_dir) / "connectors" / provider_slug
    manifest = harvest(source_dir, conn_root)
    manifest["provider"] = provider
    manifest["license_ref"] = license_ref
    # Claims with figures are the traced-figure count (a figure = a digit in
    # the claim text; the claims ledger is what trace() walks).
    claims = BrainStore(conn_root).load_claims()
    manifest["traced_figures"] = sum(
        1 for c in claims if any(ch.isdigit() for ch in str(c.get("text", "")))
    )
    main = brain.load_manifest()
    connectors = main.get("connectors") or []
    connectors = [c for c in connectors if c.get("provider") != provider]
    connectors.append({
        "provider": provider,
        "source": str(source_dir),
        "license_ref": license_ref,
        "pages": manifest["pages"],
        "claims": manifest["claims"],
        "traced_figures": manifest["traced_figures"],
        "connected_at": manifest["harvested_at"],
    })
    main["connectors"] = connectors
    brain.save_manifest(main)
    return manifest


def trace(
    figure: str, brain_dir: Path | None = None
) -> dict:
    """Trace a figure to its source record, naming the connector that supplied it."""
    result = verify(figure, brain_dir)
    if not result.get("found"):
        return result
    label = result.pop("connector", None)
    if label and label != "main":
        brain = BrainStore(brain_dir or _DEFAULT_BRAIN)
        for conn in brain.load_manifest().get("connectors") or []:
            if slugify(str(conn.get("provider", ""))) == label:
                result["connector"] = {
                    "provider": conn.get("provider"),
                    "license_ref": conn.get("license_ref"),
                }
                break
    return result


def watch(
    source_dir: Path,
    brain_dir: Path | None = None,
    every: float = 30.0,
    once: bool = False,
    on_change: Callable[[dict], None] | None = None,
) -> None:
    """Re-harvest changed sources; the brain stays current as work changes."""
    seen = {
        path: path.stat().st_mtime_ns
        for path, _ in _iter_sources(source_dir)
    }
    while True:
        time.sleep(every)
        changed = [
            path for path, _ in _iter_sources(source_dir)
            if seen.get(path) != path.stat().st_mtime_ns
        ]
        if not changed:
            continue
        for path in changed:
            seen[path] = path.stat().st_mtime_ns
        manifest = harvest(source_dir, brain_dir)
        print(f"[awbrain] re-harvested {len(changed)} changed source(s): "
              f"{manifest['pages']} pages, {manifest['claims']} claims")
        if on_change:
            on_change(manifest)
        if once:
            return
