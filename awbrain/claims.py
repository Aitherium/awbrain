"""awbrain.claims — extract claims with their evidence chain.

Deterministic on purpose: bullets, numbered items and bolded statements in a
chunk become claims; every claim records the exact file and line span that
supports it. ``verify`` walks a claim back to that span and reads the excerpt
from the SOURCE file, never from the wiki copy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .store import Claim, stable_id

_BULLET = re.compile(r"^\s*[-*]\s+(.{40,})")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.{40,})")
# Whole-line bold, OR a bold statement that OPENS a paragraph (law files put
# the claim first and the explanation after it on the same line — measured
# 2026-08-26, "Gate the changed lines..." at law-01:44 was missed whole-line).
_BOLD = re.compile(r"^\s*\*\*(.{40,}?)\*\*\s*($|[^\n])")
# A figure: currency, an explicit scale word (million/billion/percent), or a
# bare numeric unit (GB/MB/x). Prose sentences carrying one become claims —
# the shape the demo exposed: a 371-line prose report yielded 4 bullet claims
# while its figures ("1.87x faster", "~9.3 GB") sat in sentences.
_FIGURE = re.compile(
    r"(?:\$\s?\d[\d,]*|\d[\d,]*\s?(?:million|billion|trillion|percent|%|GB|TB|MB|x))",
    re.IGNORECASE,
)
_CLAIM_MAX = 220
_PROSE_FIGURE_CAP = 2  # per chunk; figures in prose stay rare, not floody


@dataclass
class _LineClaim:
    text: str
    line_no: int


def extract_claims(
    text: str, source: str, start_line: int, wiki: str
) -> list[Claim]:
    """Claims from one chunk: the strongest lines, capped and deduped.

    ``start_line`` is the chunk's first line in the source file, so spans are
    REAL line numbers a human can open."""
    seen: set[str] = set()
    claims: list[Claim] = []
    prose_figures = 0
    for line_no, line in enumerate(text.splitlines(), start=start_line):
        if len(claims) >= 5:
            break
        found = None
        for pat in (_BOLD, _BULLET, _NUMBERED):
            m = pat.match(line)
            if m:
                found = m.group(1).strip()
                break
        if found is None:
            # Prose pass: every sentence on the line carrying a figure is a
            # claim — the shape the demo exposed (a prose report's figures sat
            # in sentences, not bullets). Capped so figure-dense prose cannot
            # flood the ledger.
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                if prose_figures >= _PROSE_FIGURE_CAP or len(claims) >= 5:
                    break
                if not _FIGURE.search(sentence):
                    continue
                if _add_claim(claims, seen, source, line_no, sentence, wiki):
                    prose_figures += 1
            continue
        _add_claim(claims, seen, source, line_no, found, wiki)
    return claims


def _add_claim(
    claims: list[Claim], seen: set[str], source: str,
    line_no: int, text: str, wiki: str,
) -> bool:
    """Clean and ledger one claim; returns True when it was added."""
    # Strip markdown emphasis so the ledger carries the claim, not the
    # source's `**` markers (measured live 2026-08-26: bullet-bold
    # combinations kept "**" inside claim text).
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\s+", " ", text).strip(" *-")
    if not (30 <= len(text) <= _CLAIM_MAX):
        return False
    key = text.lower()
    if key in seen:
        return False
    seen.add(key)
    claims.append(Claim(
        id=stable_id(source, str(line_no), text),
        text=text,
        source=source,
        span=f"{line_no}-{line_no}",
        excerpt=text,
        wiki=wiki,
    ))
    return True
