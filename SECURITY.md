# Security Policy

## Supported versions

This is a research codebase. Security fixes are applied to the latest release
on `main` only.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately via
[GitHub Security Advisories](https://github.com/DiogoRibeiro7/drift-or-shift/security/advisories/new),
or by email to <hansolo.dj@gmail.com>.

Include where possible:

- a description of the issue and its impact,
- steps or a minimal script to reproduce it,
- the affected version or commit.

You can expect an acknowledgement within 7 days and a status update within 30 days.

## Scope

This project loads YAML configuration and reads/writes result artifacts from
local paths. Treat configuration files and `results/` trees as trusted input:
they are not sandboxed, and `yaml.safe_load` is used but paths are not
validated against traversal. Do not run the pipeline against untrusted
configuration from third parties.
