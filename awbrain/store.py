"""awbrain.store — the brain/ layout: a filesystem of linked markdown.

    brain/
      wiki/<slug>.md     one page per chunk: frontmatter (source, span, hash),
                         body, [[wikilinks]], backlinks section
      claims.yaml        every claim with its evidence chain
      index.json         the embed index (chunk id, slug, text, embedding)
      manifest.json      receipts: sources/pages/claims counts, embedder mode,
                         context bytes vs total bytes (the "never load
                         everything" measurement)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 60) -> str:
    slug = SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len].rstrip("-") or "page"


def stable_id(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    id: str
    slug: str
    text: str
    source: str
    start: int
    end: int
    heading: str = ""
    emb: list[float] = field(default_factory=list)
    related: list[str] = field(default_factory=list)


@dataclass
class Claim:
    id: str
    text: str
    source: str
    span: str
    excerpt: str
    wiki: str


def frontmatter(chunk: Chunk) -> str:
    return (
        "---\n"
        f"id: {chunk.id}\n"
        f"source: {chunk.source}\n"
        f"span: {chunk.start}-{chunk.end}\n"
        f"heading: {chunk.heading or '(none)'}\n"
        "---\n"
    )


def page_markdown(chunk: Chunk) -> str:
    head = f"# {chunk.heading or chunk.slug}\n" if chunk.heading else ""
    body = chunk.text.strip()
    links = "\n".join(f"- [[{slug}]]" for slug in chunk.related) if chunk.related else ""
    related = f"\n\n## Related\n{links}" if links else ""
    return f"{frontmatter(chunk)}\n{head}{body}{related}\n"


def write_claim(claim: Claim) -> dict[str, Any]:
    return {
        "id": claim.id,
        "text": claim.text,
        "evidence": {
            "file": claim.source,
            "span": claim.span,
            "excerpt": claim.excerpt[:300],
        },
        "wiki": claim.wiki,
    }


class BrainStore:
    """Read/write the brain/ directory tree."""

    def __init__(self, brain_dir: Path) -> None:
        self.root = Path(brain_dir)
        self.wiki = self.root / "wiki"
        self.claims_file = self.root / "claims.yaml"
        self.index_file = self.root / "index.json"
        self.manifest_file = self.root / "manifest.json"

    def ensure(self) -> None:
        self.wiki.mkdir(parents=True, exist_ok=True)

    def save_page(self, chunk: Chunk) -> None:
        (self.wiki / f"{chunk.slug}.md").write_text(
            page_markdown(chunk), encoding="utf-8"
        )

    def save_index(self, chunks: list[Chunk]) -> None:
        payload = [
            {"id": c.id, "slug": c.slug, "text": c.text,
             "emb": c.emb, "source": c.source, "related": c.related}
            for c in chunks
        ]
        self.index_file.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def load_index(self) -> list[Chunk]:
        if not self.index_file.is_file():
            return []
        try:
            data = json.loads(self.index_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        out = []
        for row in data:
            try:
                out.append(Chunk(
                    id=row["id"], slug=row["slug"], text=row["text"],
                    source=row.get("source", "?"), start=0, end=0,
                    emb=row.get("emb") or [],
                    related=row.get("related") or [],
                ))
            except KeyError:
                continue
        return out

    def save_claims(self, claims: list[Claim]) -> None:
        def _q(value: str) -> str:
            # JSON string escaping is valid YAML double-quoted-scalar escaping;
            # a bare quoted excerpt breaks the ledger on its first embedded
            # quote (measured 2026-08-26: law-11's excerpt contains them).
            return json.dumps(str(value), ensure_ascii=False)

        rows = "\n".join(
            f"- id: {c.id}\n  text: {_q(c.text)}\n  evidence:\n"
            f"    file: {c.source}\n    span: {c.span}\n"
            f"    excerpt: {_q(c.excerpt[:300])}\n  wiki: {c.wiki}"
            for c in claims
        )
        header = (
            "# awbrain claims ledger\n"
            "# every claim links to the file and line span that supports it.\n"
        )
        self.claims_file.write_text(header + rows + "\n", encoding="utf-8")

    def load_claims(self) -> list[dict[str, Any]]:
        if not self.claims_file.is_file():
            return []
        try:
            import yaml  # type: ignore[import-not-found]
            return yaml.safe_load(self.claims_file.read_text(encoding="utf-8")) or []
        except ImportError:
            return self._load_claims_naive()
        except yaml.YAMLError:  # type: ignore[name-defined]
            return []

    def _load_claims_naive(self) -> list[dict[str, Any]]:
        # yaml may be absent in a minimal install; parse the fixed layout.
        out: list[dict[str, Any]] = []
        cur: dict[str, Any] = {}
        for line in self.claims_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("- id: "):
                if cur:
                    out.append(cur)
                cur = {"id": line[6:]}
            elif line.startswith("  text: ") and cur:
                cur["text"] = line[8:]
            elif line.startswith("    file: ") and cur:
                cur.setdefault("evidence", {})["file"] = line[10:]
            elif line.startswith("    span: ") and cur:
                cur.setdefault("evidence", {})["span"] = line[10:]
        if cur:
            out.append(cur)
        return out

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_file.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_file.is_file():
            return {}
        try:
            return json.loads(self.manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
