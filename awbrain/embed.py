"""awbrain.embed — embeddings with an honest fallback.

Primary: the fleet router (OpenAI-compatible /v1/embeddings, nomic-embed-text)
through a trusted CA — never verify=False. Fallback: a deterministic lexical
TF vector, so a stranger with no router still gets ``ask`` (weaker recall,
zero models, recorded honestly in the manifest as ``embedder: lexical``).
"""

from __future__ import annotations

import math
import os
import re
import sys
import unicodedata
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover - exercised when httpx is absent
    httpx = None  # type: ignore[assignment]

DEFAULT_EMBED_URL = "https://127.0.0.1:8150/v1/embeddings"
DEFAULT_MODEL = "nomic-embed-text"
_CA_CANDIDATES = (
    Path(r"C:/AitherOS-Data/Library/Data/tls/ca-chain.pem"),
    Path("/app/AitherOS/Library/Data/tls/ca-chain.pem"),
    Path("/etc/aither/tls/ca-chain.pem"),
)
_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_STOP = frozenset(
    "a an and are as at be by for from has have he her his i in is it its "
    "of on or our she that the their them they this to was we were will with "
    "you your not but what which when where who how".split()
)


def _tokens(text: str) -> list[str]:
    out = []
    for raw in _TOKEN_RE.findall(unicodedata.normalize("NFKD", text.lower())):
        if raw not in _STOP and len(raw) > 1:
            out.append(raw)
    return out


def resolve_ca() -> str | None:
    for cand in _CA_CANDIDATES:
        if cand.is_file():
            return str(cand)
    return None


class Embedder:
    """Embed texts; falls back to lexical TF when the router is unreachable."""

    def __init__(
        self,
        url: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 10.0,
    ) -> None:
        self.url = url or os.environ.get("AWBRAIN_EMBED_URL", DEFAULT_EMBED_URL)
        self.model = model
        self.timeout = timeout
        self.mode = "router"  # flipped to "lexical" after a failed router probe
        self._lex_failures = 0

    # -- router ----------------------------------------------------------
    def _router_embed(self, texts: list[str]) -> list[list[float]] | None:
        if httpx is None:
            return None
        try:
            rows = self._router_post(texts)
            if rows is not None:
                return rows
        except Exception as exc:
            # The batch path is best-effort; the single-string retry below is
            # the real contract (the fleet router rejects a list input).
            print(f"[awbrain] router batch embed failed ({type(exc).__name__}); "
                  f"retrying per item", file=sys.stderr)
        # The fleet router's /v1/embeddings takes a STRING input, not a list
        # (measured 2026-08-26: a list is a 422). Retry one text per call.
        out: list[list[float]] = []
        try:
            for text in texts:
                rows = self._router_post([text])
                if rows is None:
                    return None
                out.append(rows[0])
            return out
        except Exception:
            return None

    def _router_post(self, texts: list[str]) -> list[list[float]] | None:
        payload: dict = {"model": self.model}
        if len(texts) == 1:
            payload["input"] = texts[0]
        else:
            payload["input"] = texts
        resp = httpx.post(
            self.url,
            json=payload,
            timeout=self.timeout,
            verify=resolve_ca() or True,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        rows = sorted(
            data.get("data", []), key=lambda r: int(r.get("index", 0))
        )
        return [r.get("embedding") or [] for r in rows]

    # -- lexical fallback -------------------------------------------------
    def _lex_embed(self, texts: list[str]) -> list[list[float]]:
        # TF vector over the corpus's own vocabulary, l2-normalised.
        vocab: dict[str, int] = {}
        for text in texts:
            for tok in _tokens(text):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        vecs: list[list[float]] = []
        for text in texts:
            vec = [0.0] * len(vocab)
            for tok in _tokens(text):
                if tok in vocab:
                    vec[vocab[tok]] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vecs.append([v / norm for v in vec])
        return vecs

    # -- public ----------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.mode == "router":
            got = self._router_embed(texts)
            if got is not None and len(got) == len(texts):
                return got
            self.mode = "lexical"
            self._lex_failures += 1
        return self._lex_embed(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0
