# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Email the maintainers privately with a description
and, if possible, a reproduction. We will acknowledge within 72 hours and aim for a fix + disclosure within 90 days.

## Threat model (summary)

dFactory-Lab runs **locally** and shells out to heavy ML processes (`torchrun`, `scripts/moe_convertor.py`,
dataset builders). The relevant risks:

- **Arbitrary subprocess execution** - every CLI the server wraps must run with **argument arrays** (no shell
  interpolation) and **jailed paths** under managed directories. User-supplied paths/dataset files are untrusted
  input.
- **Credential storage** - Hugging Face tokens and (optional) auth secrets must be stored **hashed/encrypted at
  rest**, never logged, never committed.
- **Network exposure** - the dev server binds to localhost by default; LAN/tunnel exposure is opt-in and gated by
  a bootstrap token (see Checklist §9).
- **Artifact safety** - model checkpoints and datasets are large and untrusted; they live under git-ignored,
  sandboxed directories.

See `Checklist.md → Auth, Security & Observability` for the full set of hardening items.
