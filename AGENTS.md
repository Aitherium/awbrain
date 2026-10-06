# awbrain for agents

Read this if you are an agent (or a human) editing this package. Short on
purpose: the commands, the traps that cost a session, and where the rest lives.
Nothing here is read at runtime — it is for you.

## What this is

PyPI distribution **`awbrain`** (version in `pyproject.toml`), import package
`awbrain`, Python >= 3.10. Your history as a wiki of linked markdown — claims
pinned to the evidence they came from.

This repository is a **synced mirror** of the AitherOS monorepo (lane
`.github/workflows/sync-awbrain.yml`). Hand edits made here are overwritten on
the next sync — change the source and let the lane publish.

## Build, test, verify

```bash
python -m pytest tests -q        # the suite: 14 tests, green at v0.1.0
pip install -e .                 # editable install for developing against it
```

The suite was run from a source checkout with no prior install. The publish
lane (`publish-brick.yml`) additionally builds the wheel, installs it and
imports it — a tree that tests green can still ship a broken wheel.

## Rules that keep this useful

- **A claim without its evidence is not a claim.** The pinned-evidence link is
  the whole product; a summarisation path that drops the source pin turns the
  wiki into prose with the same confidence and no way back to the ground.
- **The output is markdown files you own.** Generated pages stay plain,
  linked, and editable by hand — a brain you cannot diff is a brain you cannot
  audit.
- **The registry drives the public surface.** This repo's README header,
  `llms.txt` and `aither-manifest.json` are generated from the ecosystem
  registry (one yaml in the AitherOS monorepo) and rewritten on every sync.
  Change the registry; do not hand-edit the generated blocks.

## Read next

- `llms.txt` — the install/use card written for an agent to execute
- `README.md` — the human front door
- `docs/` — the generated docs site source
