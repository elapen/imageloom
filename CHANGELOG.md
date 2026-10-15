# Changelog

All notable changes to ImageLoom. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [1.1.0] - 2026-10-16

### Added
- **MCP server** (`imageloom mcp`, or `python3 core/imageloom_mcp.py`): Model Context Protocol over stdio for Claude Desktop, Claude Code, Cursor, VS Code, Zed, Cline and any other MCP client. Pure standard library, no new dependency, no telemetry; protocol versions 2025-06-18, 2025-03-26 and 2024-11-05.
  Twelve tools with JSON Schemas and honest read-only / destructive / idempotent / open-world hints: `generate_image`, `edit_image`, `upscale_image`, `export_image`, `get_job`, `wait_for_job`, `cancel_job`, `list_jobs`, `list_models`, `add_model`, `set_default_model`, `recent_images`.
  Generation is asynchronous (a job id, then `wait_for_job` in slices of at most 50 s with progress notifications and cancellation), results carry absolute paths, structured data and downscaled previews, and `add_model` is a dry run unless told otherwise. See [docs/MCP.md](docs/MCP.md).

### Changed
- The macOS app and the Windows build now bundle the `imageloom` CLI and the MCP server next to the server, so MCP works from the app's own Python without a checkout.
- `core/imageloom`: model selection is now `pick_model()` (returns the reason instead of printing it; `resolve_model()` behaves as before), shared with the MCP server.

## [1.0.0] - 2026-10-14

First public release.

### Added
- Native macOS app (Apple Silicon, Swift + WKWebView, embedded Python) with a DMG; the server starts with the app and stops, with any running job, when it quits.
- Experimental Windows installer and portable zip (cross-built on macOS, **not yet run on real Windows**).
- Chat-style studio: separate chats, each with its own model and effort level; every result can become the next input; command palette (`⌘K`), new chat (`⌘N`).
- Create, edit by instruction, brush-mask edit (only painted pixels are pasted into the full-resolution original), before/after slider, version tree, favourites, gallery.
- Upscale ×2 / ×4 / 2K / 4K / 8K with Real-ESRGAN (quality / fast / illustration), optionally chained after every render.
- Export to PNG, WebP, JPEG and SVG (vtracer presets); real transparent backgrounds with Qwen-Image.
- Model manager: add a model from a Hugging Face link (family detection, file picker with sizes, companion files, resumable downloads, disk-space check, optional token for gated repos).
- Models: Qwen-Image 2.1 Turbo Q8 (default, verified), Z-Image Turbo Q4_K (verified); FLUX.1, SDXL and SD 1.5 recipes (experimental).
- Time estimates that learn real timings per model and size.
- CLI (`imageloom`): create, edit, upscale, export, `models add|use|rm`, and ready-made skills for Claude Code, Cursor and Hermes.
- English and Russian UI (auto-detected, switchable), also for the macOS and Windows hosts and the Windows installer.
- Security hardening and a 181-test offline suite (guard, path traversal, redirects that must not leak the token, archive extraction, corrupt state files).
