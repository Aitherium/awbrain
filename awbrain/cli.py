"""awbrain CLI — harvest, connect, ask, verify, trace, watch, status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import ask, connect, harvest, trace, verify, watch
from .store import BrainStore

_EPILOG = """\
commands:
  harvest <dir>          build/refresh the wiki from a folder of notes/sessions
  connect <provider>     register a data provider (--from dir, --license NAME):
                         its figures trace to the source records it came from
  ask "<question>"       answer from ONLY the top-k pages (never everything)
  verify "<claim>"       walk a claim back to the file and line that supports it
  trace "<figure>"       trace a figure to its source record, naming the
                         connector that supplied it
  watch <dir>            re-harvest changed sources so the brain stays current
  status                 brain receipts: sources, pages, claims, embedder mode

The brain is a folder of linked markdown you can open in any editor: no
proprietary blob, no vendor cloud. Embeddings come from the fleet router
(nomic-embed-text) when reachable, else a lexical fallback recorded honestly.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="awbrain",
        description="Your history as a wiki of linked markdown — claims pinned "
                    "to the evidence.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=(
        "harvest", "connect", "ask", "verify", "trace", "watch", "status"
    ))
    parser.add_argument("target", nargs="?", help="source dir (harvest/watch) or question/claim")
    parser.add_argument("--brain", default="brain", help="brain dir (default: ./brain)")
    parser.add_argument("--k", type=int, default=4, help="pages retrieved per ask")
    parser.add_argument("--every", type=float, default=30.0, help="watch poll seconds")
    parser.add_argument("--once", action="store_true", help="watch: one pass then exit")
    parser.add_argument("--from", dest="source", help="connect: source dir (dir of "
                        "files, mirrored site, or a licensed portal's export)")
    parser.add_argument("--license", dest="license_ref", help="connect: existing "
                        "license reference (stored name, never the key itself)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    brain_dir = Path(args.brain)
    if args.command == "harvest":
        if not args.target:
            parser.error("harvest needs a source dir")
        manifest = harvest(Path(args.target), brain_dir)
        _emit(manifest, args.json,
              f"harvested: {manifest['sources']} sources, {manifest['pages']} "
              f"pages, {manifest['claims']} claims (embedder: {manifest['embedder']})")
    elif args.command == "connect":
        if not args.target or not args.source:
            parser.error("connect needs a provider name and --from <dir>")
        manifest = connect(
            args.target, Path(args.source), brain_dir, license_ref=args.license_ref
        )
        _emit(manifest, args.json,
              f"connected {args.target}: {manifest['pages']} pages, "
              f"{manifest['claims']} claims, "
              f"{manifest['traced_figures']} figures traceable to source records"
              + (f" (license: {args.license_ref})" if args.license_ref else ""))
    elif args.command == "ask":
        if not args.target:
            parser.error("ask needs a question")
        result = ask(args.target, brain_dir, k=args.k)
        _emit(result, args.json, _format_answer(result))
    elif args.command == "verify":
        if not args.target:
            parser.error("verify needs a claim")
        result = verify(args.target, brain_dir)
        _emit(result, args.json, _format_verify(result))
    elif args.command == "trace":
        if not args.target:
            parser.error("trace needs a figure")
        result = trace(args.target, brain_dir)
        _emit(result, args.json, _format_trace(result))
    elif args.command == "watch":
        if not args.target:
            parser.error("watch needs a source dir")
        watch(Path(args.target), brain_dir, every=args.every, once=args.once)
    elif args.command == "status":
        store = BrainStore(brain_dir)
        manifest = store.load_manifest()
        if not manifest:
            parser.error(f"no brain at {brain_dir} — run `awbrain harvest <dir>` first")
        _emit(manifest, args.json, _format_status(manifest))
    return 0


def _emit(payload: dict, as_json: bool, human: str) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(human)


def _format_answer(result: dict) -> str:
    answer = result.get("answer")
    out = []
    if answer:
        out.append(f"ANSWER ({result['pages_retrieved']} pages: "
                   f"{result['k']} retrieved + 2-hop links, "
                   f"{result['context_bytes']} of {result['total_bytes']} bytes "
                   f"loaded — not everything):\n{answer}")
    else:
        out.append(f"NO ROUTER — retrieved {result['pages_retrieved']} pages "
                   "(lexical mode); no answer generator without a router.")
    out.append("\nSOURCES:")
    for src in result.get("sources", []):
        score = src.get("score")
        hop = f" (hop 2 via [[{src.get('page')}]])" if score is None else \
            f" (cosine {score})"
        out.append(f"  [[{src['page']}]] {src['source']}{hop}")
    return "\n".join(out)


def _format_verify(result: dict) -> str:
    if not result.get("found"):
        return f"claim not in the ledger: {result.get('claim', '')}"
    ev = result.get("evidence") or {}
    excerpt = str(ev.get("excerpt", "")).strip() or "(no excerpt)"
    return (
        f"CLAIM: {result.get('claim')}\n"
        f"EVIDENCE: {ev.get('file')} lines {ev.get('span')}\n"
        f"WIKI PAGE: [[{result.get('wiki')}]]\n"
        f"SOURCE SPAN:\n{excerpt}"
    )


def _format_trace(result: dict) -> str:
    if not result.get("found"):
        return f"figure not in the ledger: {result.get('claim', '')}"
    ev = result.get("evidence") or {}
    excerpt = str(ev.get("excerpt", "")).strip() or "(no excerpt)"
    conn = result.get("connector") or {}
    provider = f"via connector '{conn.get('provider')}'" if conn.get("provider") \
        else "(no connector)"
    license_note = f" license: {conn.get('license_ref')}" if conn.get("license_ref") \
        else ""
    return (
        f"FIGURE: {result.get('claim')}\n"
        f"TRACES TO: {ev.get('file')} lines {ev.get('span')} {provider}{license_note}\n"
        f"WIKI PAGE: [[{result.get('wiki')}]]\n"
        f"SOURCE RECORD:\n{excerpt}"
    )


def _format_status(manifest: dict) -> str:
    connectors = manifest.get("connectors") or []
    lines = [
        f"brain: {manifest.get('pages', 0)} pages from {manifest.get('sources', 0)} "
        f"sources · {manifest.get('claims', 0)} claims · "
        f"embedder: {manifest.get('embedder', '?')} · "
        f"harvested {manifest.get('harvested_at', '?')}"
    ]
    for conn in connectors:
        lines.append(
            f"  connector {conn.get('provider')}: {conn.get('pages')} pages, "
            f"{conn.get('traced_figures')} traced figures"
            + (f" · license {conn.get('license_ref')}"
               if conn.get("license_ref") else "")
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
