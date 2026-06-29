# Frontend npm Dependency Audit

**Date:** 2026-06-10
**Branch:** chore/deps-frontend
**Initial vulnerabilities:** 19 (10 moderate, 7 high, 2 critical)
**Final vulnerabilities:** 0

## Direct Dependency Updates

| Package | Old Version | New Version | Reason |
|---------|-------------|-------------|--------|
| axios | ^1.13.6 | ^1.17.0 | 23 SSRF/prototype pollution CVEs |
| i18next-http-backend | ^3.0.1 | ^3.0.5 | Path Traversal/URL Injection (GHSA-q89c-q3h5-w34g) |
| react-router-dom | ^6.30.3 | ^6.30.4 | Open redirect via protocol-relative URL (GHSA-2j2x-hqr9-3h42) |
| postcss | ^8.5.8 | ^8.5.10 | XSS via unescaped </style> in CSS output (GHSA-qx2v-qp2m-jg93) |
| vite | ^6.4.2 | ^6.4.3 | Server CVEs in picomatch sub-dependency |
| vitest | ^3.0.7 | ^3.2.6 | Critical: arbitrary file read/execution in UI server (GHSA-5xrq-8626-4rwp) |
| @vitest/coverage-v8 | ^3.0.7 | ^3.2.6 | Matches vitest parent version |

## Override Additions

Transitive dependencies pinned to safe versions via npm `overrides`:

| Package | Override Version | Vulnerability |
|---------|-----------------|---------------|
| brace-expansion | 2.1.1 | ReDoS (GHSA-v6h2-p8h4-qcjw, GHSA-f886-m6hf-6m8v, GHSA-jxxr-4gwj-5jf2) |
| braces | 3.0.3 | Uncontrolled resource consumption (GHSA-grv7-fg5c-xmjg) |
| glob | 11.1.0 | CLI command injection (GHSA-5j98-mcp5-4vw2) |
| micromatch | 4.0.8 | ReDoS (GHSA-952p-6rrq-rcjv) |
| object-path | 0.11.8 | Prototype Pollution (GHSA-cwx2-736x-mf6w, GHSA-8v63-cqqc-6r2c) |
| tmp | 0.2.6 | Path traversal via prefix/postfix (GHSA-ph9p-34f9-6g65) |
| uuid | 14.0.0 | Missing buffer bounds check (GHSA-w5hq-g745-h8pq) |
| ws | 8.21.0 | Uninitialized memory disclosure (GHSA-58qx-3vcg-4xpx) |
| tinyglobby | 0.2.17 | Pulls safe picomatch@^4.0.4 |

### Nested Overrides

| Path | Override | Reason |
|------|----------|--------|
| vite > picomatch | 4.0.4 | Picomatch@4.0.3 is vulnerable; 4.0.4 is safe |
| tinyglobby > picomatch | 4.0.4 | Ensures consistent safe picomatch within tinyglobby |

## Lodash

lodash@^4.18.1 (latest published) -- no CVEs reported in npm audit.

## Build Verification

`npm run build` succeeds. Build uses ~4GB heap; to run on constrained hardware set `NODE_OPTIONS="--max-old-space-size=4096"`.

## Notes

- sort-by@1.2.0 depends on object-path@0.6.0 which is vulnerable. The `object-path` override forces 0.11.8 tree-wide. No behavioral change expected as sort-by uses only basic get/set operations.
- i18next-cli depends on glob@11.x. The glob override to 11.1.0 (past 11.0.3 where the CLI vuln was fixed) resolves this.
- Vite 7.x is available but was not upgraded to avoid potential breakage with vite-plugin-monaco-editor and other Vite ecosystem plugins. The picomatch vulnerability in Vite 6.4.3 was resolved via nested overrides instead.
## 2023-11-20
- Removed broken `global.Intl.DateTimeFormat` mock in `web/src/utils/dateUtil.test.ts` to fix 7 failing tests and properly let format function run.
- Fixed i18n mock in `web/src/utils/dateUtil.test.ts` to return string key properly, resolving failing test for 12hour AM/PM formatting.

## General Status update on 2023-11-20
- Unit tests are mostly succeeding. `web/src/utils/dateUtil.test.ts` issues have been resolved. Backend tests are experiencing mocking issues in `test_runner.py` affecting 60+ tests, which will be logged to `Jules/improvements.md` for future implementation.
