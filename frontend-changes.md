# Frontend Changes — Dark/Light Theme Toggle

## Overview
Added a theme toggle button that allows users to switch between dark and light themes with smooth transitions, localStorage persistence, and accessible design.

## Files Modified

### `frontend/index.html`
- Added theme toggle button (`#themeToggle`) before the container div
- Button contains both sun and moon SVG icons with smooth crossfade/rotation transitions
- Sun icon visible in dark mode, moon icon visible in light mode
- Accessible: `aria-label` updates dynamically, keyboard focusable via `focus-visible`
- Updated CSS and JS cache-busting versions

### `frontend/style.css`
- Restructured `:root` CSS variables with explicit dark theme defaults
- Added `[data-theme="light"]` selector block with light theme variables:
  - Background: `#f8fafc`, Surface: `#ffffff`, Text: `#0f172a`
  - Adjusted border, link, shadow, and surface colors for light context
- Added `transition` on `body` and key structural elements for smooth theme switching (0.3s ease)
- Added `.theme-toggle` styles: fixed top-right position (z-index 100), circular button with hover/focus/active states
- Icon swap animation: sun rotates out while moon rotates in (and vice versa)

### `frontend/script.js`
- Added `initTheme()` called on DOMContentLoaded
- `applyTheme(theme)` sets `data-theme` attribute on `<html>` and updates `aria-label`
- Toggle button click handler switches between "dark" and "light"
- Theme preference persisted in `localStorage`
- Default theme is "dark" when no saved preference exists

## Theme Colors

| Variable | Dark (default) | Light |
|---|---|---|
| `--background` | `#0f172a` | `#f8fafc` |
| `--surface` | `#1e293b` | `#ffffff` |
| `--surface-hover` | `#334155` | `#f1f5f9` |
| `--text-primary` | `#f1f5f9` | `#0f172a` |
| `--text-secondary` | `#94a3b8` | `#475569` |
| `--border-color` | `#334155` | `#e2e8f0` |
| `--link-color` | `#60a5fa` | `#2563eb` |
| `--shadow` | `rgba(0,0,0,0.3)` | `rgba(0,0,0,0.08)` |
