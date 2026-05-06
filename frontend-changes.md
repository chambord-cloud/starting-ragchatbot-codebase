# Frontend Code Quality Tooling

## Overview

Added code quality tooling to the frontend with Prettier for formatting and ESLint for linting.

## Files Added

| File | Purpose |
|------|---------|
| `package.json` | npm project with format/lint/check/fix scripts |
| `.prettierrc` | Prettier configuration (2-space indent, single quotes, semicolons, trailing commas) |
| `.prettierignore` | Excludes backend, docs, Python files from formatting |
| `eslint.config.js` | ESLint flat config targeting `frontend/**/*.js` with browser globals |

## Files Modified

| File | Change |
|------|--------|
| `frontend/index.html` | Formatted with Prettier |
| `frontend/script.js` | Formatted with Prettier; removed unused `i` parameter in `.map()` callback |
| `frontend/style.css` | Formatted with Prettier |

## npm Scripts

| Script | Command | Purpose |
|--------|---------|---------|
| `format` | `prettier --write "frontend/**/*.{html,css,js}"` | Auto-format all frontend files |
| `format:check` | `prettier --check "frontend/**/*.{html,css,js}"` | Check formatting (CI-safe) |
| `lint` | `eslint "frontend/**/*.js"` | Lint JavaScript files |
| `lint:fix` | `eslint --fix "frontend/**/*.js"` | Auto-fix lint issues |
| `check` | `npm run format:check && npm run lint` | Run all quality checks |
| `fix` | `npm run format && npm run lint:fix` | Auto-fix all formatting and lint issues |

## ESLint Rules

- Extends `@eslint/js` recommended config
- `ecmaVersion: "latest"` with browser globals (`document`, `window`, `console`, `fetch`, `marked`)
- `no-unused-vars`: warn (with `_` prefix ignore pattern)
- `no-undef`: error
- `no-console`: off
- `no-constant-condition`: warn
