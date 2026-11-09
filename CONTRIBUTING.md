# Contributing to Yamabiko

Yamabiko is narrowly focused on real-time Japanese subtitles for Windows and
NVIDIA GPUs. Issues are welcome; pull requests are reviewed case-by-case.

## Before you open a PR

- Read [CLA.md](CLA.md). Every commit must carry a `Signed-off-by:`
  trailer (DCO v1.1); `git commit -s` adds this automatically. The
  trailer certifies you have the right to submit the contribution.
- Include the line `I agree to the Yamabiko CLA.` in the PR
  description. This records explicit assent to the license grant in
  CLA §2, which the DCO alone does not cover.
- Open an issue first for non-trivial changes. A quick sketch of the
  idea saves both of us time if the direction isn't right for the
  project.

## Dev environment

See [README.md](README.md) § "Set up a development environment".

For installation help or ordinary usage questions, see the README's
[troubleshooting section](README.md#fix-common-problems) or email
[support@kaze.be](mailto:support@kaze.be).

## Scope

This project is narrowly focused on real-time Japanese subtitle
generation for Windows + NVIDIA GPU. PRs that broaden the scope
(additional languages, cloud hosting, mobile, etc.) are unlikely
to be merged. PRs that improve the core experience (better sentence
boundaries, smoother overlay rendering, smaller install footprint,
better packaging) are very welcome.

## Testing

```powershell
.venv/Scripts/python.exe -m pytest -q
```

PRs must keep the suite green. New behavior needs new tests.

Lint the files you change (rule set pinned in `pyproject.toml`):

```powershell
.venv/Scripts/python.exe -m ruff check <changed paths>
```

The repo-wide lint baseline is not clean yet, so lint changed paths
rather than the whole tree.

All project spaces follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Design docs

Subsystem design notes live in [docs/design/](docs/design/). Read the
relevant one before proposing a significant change to that subsystem.
It covers the current approach, the constraints it's working under,
and the tradeoffs already considered.

## Releasing (maintainers)

The tag-and-publish workflow lives in [docs/RELEASE.md](docs/RELEASE.md).
Run `scripts/preflight.py` before any `v*` tag; it exercises the
external touchpoints (HF repos, download signatures, audio stack,
PyInstaller spec) that mocked unit tests don't cover.
