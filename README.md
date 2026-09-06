# awbrain

Your history as a wiki of linked markdown — every claim pinned to the
file and line that produced it.

Every AI memory system sells you a black box: sessions vanish into a vendor
cloud, and "it remembers" means "the vendor's model was prompted with a blob".
Memory should be files you can read, claims you can check against the session
or file that produced them, and retrieval that never dumps your whole history
into a prompt.

**awbrain** is a folder of linked markdown you can open in any editor: no
proprietary blob, no vendor cloud. Harvest your notes and sessions into a
wiki, ask questions against only the top-k pages (never everything), and walk
any claim back to the source line that supports it.

## Install

```bash
pip install awbrain
```

## Use

```bash
# build (or refresh) the wiki from a folder of notes/sessions
awbrain harvest ~/notes

# answer from ONLY the top-k pages — never your whole history
awbrain ask "what did we decide about the pool?"

# walk a claim back to the file and line that supports it
awbrain verify "a self-owned platform beats a subscription computer"

# re-harvest changed sources so the brain stays current
awbrain watch ~/notes

# receipts: sources, pages, claims, embedder mode
awbrain status
```

## How it works

- `harvest` reads a folder of markdown/notes and writes a linked wiki:
  pages with `[[wikilinks]]`, each claim pinned to its source file and line
  span.
- `ask` retrieves top-k pages plus a two-hop wikilink expansion — never the
  whole history — and answers with `[[citation]]` markers.
- `verify` walks a claim back to the source file and line span that supports
  it, or says it cannot find the evidence.
- `watch` re-harvests changed sources so the brain stays current as work
  changes.

Embeddings come from a router (nomic-embed-text) when reachable, else an
honest lexical fallback — the manifest records which mode was used, so the
retrieval story is never overstated.

## Compose with

- `awgraph` — the wikilinks ARE a graph; `ask` already walks two hops.
- `awm` — claims as scoped memory.
- `awseal` — failures as "what not to do" claims.

MIT licensed.
