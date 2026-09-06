"""awbrain tests — the loop must be provable offline, deterministically."""

from __future__ import annotations

from pathlib import Path

import pytest
from awbrain.claims import extract_claims
from awbrain.engine import ask, connect, harvest, trace, verify
from awbrain.store import BrainStore, slugify, stable_id

FIXTURE = """\
# The Platform

## Ownership

- A self-owned platform beats a subscription computer.
- The pooled frontier model is sharded across your own machines.

## The Letter

The letter from the machine explains what a self-owned platform actually is.

**Ownership is not a feature; it is the product.**

1. Models run on hardware you own.
2. Memory is a folder of linked markdown you can read.
"""


@pytest.fixture()
def brain_dir(tmp_path: Path) -> Path:
    src = tmp_path / "notes"
    src.mkdir()
    (src / "platform.md").write_text(FIXTURE, encoding="utf-8")
    out = tmp_path / "brain"
    harvest(src, out)
    return out


def test_slugify_and_ids() -> None:
    assert slugify("The Letter from the Machine!") == "the-letter-from-the-machine"
    assert len(stable_id("a", "1", "b")) == 16


def test_harvest_builds_wiki_with_links(brain_dir: Path) -> None:
    pages = list((brain_dir / "wiki").glob("*.md"))
    assert len(pages) >= 3  # heading-level chunks
    joined = "\n".join(p.read_text(encoding="utf-8") for p in pages)
    assert "## Related" in joined  # link pass ran
    assert "[[platform-ownership]]" in joined or "[[platform-the-letter]]" in joined


def test_harvest_writes_claims_ledger(brain_dir: Path) -> None:
    claims = BrainStore(brain_dir).load_claims()
    texts = [c["text"] for c in claims]
    assert any("self-owned platform beats" in t for t in texts)
    assert any("Ownership is not a feature" in t for t in texts)
    ev = claims[0]["evidence"]
    assert "file" in ev and "span" in ev  # every claim pins its evidence


def test_claims_have_real_spans() -> None:
    claims = extract_claims(FIXTURE, "platform.md", 1, "x")
    spans = [c.span for c in claims]
    assert all(int(s.split("-")[0]) > 0 for s in spans)


def test_prose_figures_become_claims(tmp_path: Path) -> None:
    """The density gap the live demo exposed: a prose report's figures sat in
    sentences, not bullets, so the ledger missed them. Sentences carrying a
    figure are claims now, capped so figure-dense prose cannot flood."""
    src = tmp_path / "notes"
    src.mkdir()
    (src / "report.md").write_text(
        "# Report\n\n## Q3\n\n"
        "Revenue grew to $420 million in Q3. The drafter model makes it 1.87x faster.\n"
        "No figure here, just context.\n",
        encoding="utf-8",
    )
    out = tmp_path / "brain"
    harvest(src, out)
    claims = BrainStore(out).load_claims()
    texts = [c["text"] for c in claims]
    assert any("420 million" in t for t in texts)
    assert any("1.87x" in t for t in texts)
    assert not any("**" in t for t in texts)  # the ledger never carries markers


def test_claim_text_has_no_markdown_markers(brain_dir: Path) -> None:
    claims = BrainStore(brain_dir).load_claims()
    assert not any("**" in c["text"] for c in claims)


def test_subchunked_section_spans_point_at_the_real_line(
    tmp_path: Path,
) -> None:
    """A long section split at a paragraph boundary must pin a claim in the
    SECOND paragraph to its REAL line — the old para_start arithmetic shifted
    every later span down, so a figure claimed line 82 while its bullet sat
    on a later line (measured live 2026-08-26: traced to an empty line)."""
    src = tmp_path / "notes"
    src.mkdir()
    # Section >700 chars with a paragraph break, then a figure bullet.
    para1 = " ".join(["word" * 12] * 22)  # ~264 chars
    para2 = (" ".join(["word"] * 40))  # ~200 chars
    text = (
        "# Long\n\n"
        "## Section\n\n"
        f"{para1}\n\n"
        f"{para2}\n\n"
        "- Revenue grew to $420 million in Q3, up from $310 million a year earlier.\n"
    )
    lines = text.splitlines()
    (src / "long.md").write_text(text, encoding="utf-8")
    out = tmp_path / "brain"
    harvest(src, out)
    claims = BrainStore(out).load_claims()
    bullet = [c for c in claims if "420 million" in c["text"]][0]
    span = bullet["evidence"]["span"]
    start, _, _ = span.partition("-")
    line_no = int(start)
    real_line = lines[line_no - 1]
    assert "420 million" in real_line  # the span IS the bullet's real line
    result = verify("420 million", out)
    assert "420 million" in (result["evidence"].get("excerpt") or "")


def test_ask_lexical_retrieves_only_topk(brain_dir: Path) -> None:
    result = ask("does ownership beat a subscription?", brain_dir, k=2)
    assert result["k"] == 2
    # The never-load-everything invariant is the BOUND, not a byte compare:
    # on a tiny brain everything fits, on a real one the cap holds.
    assert 2 <= result["pages_retrieved"] <= result["k"] + 4  # capped 2-hop
    slugs = {s["page"] for s in result["sources"]}
    assert slugs  # something was retrieved
    assert any(s.get("hop") == 2 for s in result["sources"]) or \
        len(result["sources"]) == result["pages_retrieved"]


def test_verify_walks_claim_to_evidence(brain_dir: Path) -> None:
    result = verify("self-owned platform beats a subscription", brain_dir)
    assert result["found"] is True
    ev = result["evidence"]
    assert ev["file"] == "platform.md"  # relative to the harvested source root
    assert "self-owned platform beats" in ev["excerpt"].lower()


def test_verify_unknown_claim_is_honest(brain_dir: Path) -> None:
    result = verify("the moon is made of cheese", brain_dir)
    assert result["found"] is False


def test_manifest_receipts(brain_dir: Path) -> None:
    m = BrainStore(brain_dir).load_manifest()
    assert m["pages"] >= 3
    assert m["claims"] >= 3
    assert m["embedder"] in ("router", "lexical")
    assert m["total_bytes"] > 0


def test_connect_registers_connector_and_traces_figures(
    brain_dir: Path, tmp_path: Path
) -> None:
    """The licensed-source shape: a provider's export becomes a connector whose
    figures trace to the source records — the 'every figure traces to the
    source record it came from' promise, keyless."""
    src = tmp_path / "provider-export"
    src.mkdir()
    (src / "market.md").write_text(
        "# Market\n\n## Q3\n\n"
        "- Revenue grew to $420 million in Q3, up from $310 million a year earlier.\n"
        "- Operating margin reached 18 percent, the highest level in five years.\n",
        encoding="utf-8",
    )
    manifest = connect("DunAndBradstreet", src, brain_dir, license_ref="d&b-license-1")
    assert manifest["provider"] == "DunAndBradstreet"
    assert manifest["license_ref"] == "d&b-license-1"
    assert manifest["traced_figures"] == 2  # both claims carry digits

    store = BrainStore(brain_dir)
    main = store.load_manifest()
    conns = [c for c in main.get("connectors", []) if c["provider"] == "DunAndBradstreet"]
    assert conns and conns[0]["traced_figures"] >= 2

    # The figure traces back through the connector to the source record.
    result = trace("420 million", brain_dir)
    assert result["found"] is True
    assert result["connector"]["provider"] == "DunAndBradstreet"
    assert result["connector"]["license_ref"] == "d&b-license-1"
    assert "420 million" in (result["evidence"].get("excerpt") or "").lower()

    # Re-connecting the same provider replaces, never duplicates.
    connect("DunAndBradstreet", src, brain_dir)
    conns2 = [c for c in store.load_manifest().get("connectors", [])
              if c["provider"] == "DunAndBradstreet"]
    assert len(conns2) == 1
    assert conns2[0]["license_ref"] is None


def test_trace_unknown_figure_is_honest(brain_dir: Path) -> None:
    result = trace("1,000,000 widgets", brain_dir)
    assert result["found"] is False


def test_ask_searches_connected_providers(tmp_path: Path) -> None:
    """A fresh brain holding only a connected provider must answer from the
    connector's index — the data-terminal contract: ask works over connected
    sources, no separate login to reach them."""
    src = tmp_path / "provider-export"
    src.mkdir()
    (src / "market.md").write_text(
        "# Market\n\n## Q3\n\n"
        "- Revenue grew to $420 million in Q3, up from $310 million a year earlier.\n"
        "- Operating margin reached 18 percent, the highest level in five years.\n",
        encoding="utf-8",
    )
    brain = tmp_path / "brain"
    connect("MarketData", src, brain)
    result = ask("how much revenue was reported in Q3?", brain, k=2)
    assert result["pages_retrieved"] >= 1  # the connector's pages were retrieved
    assert any(s["source"] == "market.md" for s in result["sources"])
    assert result["total_bytes"] > 0
