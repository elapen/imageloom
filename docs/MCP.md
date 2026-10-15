# ImageLoom as an MCP server

ImageLoom ships a [Model Context Protocol](https://modelcontextprotocol.io) server, so any MCP client (Claude Desktop, Claude Code, Cursor, VS Code, Zed, Cline, ...)
can **generate, edit, upscale and export images and manage models, locally**, through the ImageLoom you already have. No cloud, no API key, no extra dependency:

- one file, [`core/imageloom_mcp.py`](../core/imageloom_mcp.py), pure Python 3.9+ standard library;
- stdio transport: JSON-RPC 2.0 on stdin/stdout (one message per line), logs on stderr;
- its only network peer is the ImageLoom server on `127.0.0.1`, which it starts on demand exactly like the `imageloom` CLI (it reuses the CLI's code). A server started this way keeps running after your client quits,
  so a long job is not lost; `imageloom stop` stops it (and any running job).

Start it with either of these (they are the same program):

```bash
python3 /ABS/PATH/TO/imageloom/core/imageloom_mcp.py
python3 /ABS/PATH/TO/imageloom/core/imageloom mcp          # or `imageloom mcp` if the CLI is on your PATH
```

Run by hand it just waits for a client on stdin (Ctrl-D quits). You normally never run it yourself: you add it to your client's configuration.

> **Requirements.** A working ImageLoom (the app's setup screen, or `imageloom ui`, has downloaded the engine and a model: see the [README](../README.md)) and Python 3.9 or newer.
> From version 1.1.0 the **desktop app bundles the MCP server and its own Python**, so you do not need a checkout or a system Python:
> macOS: `/Applications/ImageLoom.app/Contents/Resources/python/bin/python3` + `/Applications/ImageLoom.app/Contents/Resources/core/imageloom_mcp.py`;
> Windows (installer): `<install folder>\python\python.exe` + `<install folder>\app\core\imageloom_mcp.py`. From a git checkout use your own Python and `core/imageloom_mcp.py`.
> The server finds the running app's ImageLoom by itself.
> Previews of finished pictures need Pillow (bundled with the app; otherwise `pip install pillow`) **or** macOS (the built-in `sips` is used); without either you still get every file path.

## Install in your client

Use **absolute paths** (clients start the server from an unpredictable folder) and, if you can, an absolute Python too (`/usr/bin/python3`, `/opt/homebrew/bin/python3`):
GUI apps often have a shorter `PATH` than your terminal. `/ABS/PATH/TO/imageloom` below is your checkout of this repository (or the app's `Contents/Resources` folder, see Requirements: swap in the bundled `python3` for the `command` and `core/imageloom_mcp.py` inside the app for the script).

The formats below follow each client's documentation at the time of writing. They were not all tried against the real clients (see [Limitations](#limitations)), so if your client's version has a different schema, keep the three facts that matter: *command*, *arguments*, optional *environment*.

### Claude Desktop

Edit `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`), then quit and reopen Claude:

```json
{
  "mcpServers": {
    "imageloom": {
      "command": "/usr/bin/python3",
      "args": ["/ABS/PATH/TO/imageloom/core/imageloom_mcp.py"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add imageloom -- python3 /ABS/PATH/TO/imageloom/core/imageloom mcp
claude mcp add --scope user imageloom -- python3 /ABS/PATH/TO/imageloom/core/imageloom mcp   # for every project
```

or commit a `.mcp.json` next to your project:

```json
{ "mcpServers": { "imageloom": { "command": "python3", "args": ["/ABS/PATH/TO/imageloom/core/imageloom", "mcp"] } } }
```

### Cursor

`.cursor/mcp.json` in a project, or `~/.cursor/mcp.json` for all projects:

```json
{
  "mcpServers": {
    "imageloom": { "command": "python3", "args": ["/ABS/PATH/TO/imageloom/core/imageloom_mcp.py"] }
  }
}
```

### VS Code (GitHub Copilot agent mode)

`.vscode/mcp.json` in a workspace (or *MCP: Add Server* in the command palette):

```json
{
  "servers": {
    "imageloom": { "type": "stdio", "command": "python3", "args": ["/ABS/PATH/TO/imageloom/core/imageloom_mcp.py"] }
  }
}
```

### Zed

`settings.json` (Zed calls MCP servers *context servers*):

```json
{
  "context_servers": {
    "imageloom": { "source": "custom", "command": "python3", "args": ["/ABS/PATH/TO/imageloom/core/imageloom_mcp.py"], "env": {} }
  }
}
```

### Cline (VS Code)

*MCP Servers → Configure* opens `cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "imageloom": { "command": "python3", "args": ["/ABS/PATH/TO/imageloom/core/imageloom_mcp.py"], "disabled": false }
  }
}
```

### Any other client (generic stdio)

Command `python3`, arguments `["/ABS/PATH/TO/imageloom/core/imageloom_mcp.py"]`, no URL, no authentication. On Windows write `"command": "python"` (or `py`) and escape backslashes in JSON: `"C:\\Users\\you\\imageloom\\core\\imageloom_mcp.py"`.

### Environment variables (all optional)

| Variable | Meaning |
|---|---|
| `IMAGELOOM_HOME` | Use another data folder (models, history, settings). Handy for trying things without touching your real one. |
| `IMAGELOOM_PORT` | Port of the ImageLoom server. Normally not needed: the running app's port is found through `server.json`. |
| `IMAGELOOM_SD_CLI` | Use your own `sd-cli` engine binary. |
| `IMAGELOOM_MCP_DEBUG=1` | Log every request to stderr (your client usually keeps that log; Claude Desktop: `~/Library/Logs/Claude/mcp-server-imageloom.log`). |

Put them in the client's `"env": { ... }` block.

## Tools

| Tool | What it does | Needs the engine | Time |
|---|---|---|---|
| [`generate_image`](#generate_image) | Text to image. Submits a **background job**, returns a `job_id`. | yes | ~4 min per 768 px image (base M4) |
| [`edit_image`](#edit_image) | Change a picture by instruction (up to 2 extra reference pictures). Job. | yes | ~2.5 min at the default 512 px tier |
| [`upscale_image`](#upscale_image) | AI upscale to 2x / 4x / 2K / 4K / 8K. Job. | yes (+ upscaler weights) | seconds to minutes |
| [`export_image`](#export_image) | Convert to PNG / WebP / JPEG or trace to SVG. Immediate. | no | seconds |
| [`get_job`](#get_job) | Status, progress, ETA, and the finished paths + previews of a job. | no | instant |
| [`wait_for_job`](#wait_for_job) | Block (at most 50 s per call) until a job ends; sends progress notifications. | no | up to 50 s |
| [`cancel_job`](#cancel_job) | Cancel a queued or running job. | no | instant |
| [`list_jobs`](#list_jobs) | The jobs the server remembers (shared with the app and the CLI). | no | instant |
| [`list_models`](#list_models) | Installed models with status, size, abilities (`edit`, `alpha`), default. | no | instant |
| [`add_model`](#add_model) | Plan (default) or start the download of a model from a Hugging Face link. | no | plan: seconds; download: minutes to hours, in the background |
| [`set_default_model`](#set_default_model) | Choose the default model (also changes the app's picker). | no | instant |
| [`recent_images`](#recent_images) | The newest pictures in the gallery, with paths, prompts and optional previews. | no | instant |

Every tool has a JSON Schema (`tools/list`) and annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`):

| Hints | Tools |
|---|---|
| read-only | `get_job`, `wait_for_job`, `list_jobs`, `list_models`, `recent_images` |
| creates new files, never overwrites | `generate_image`, `edit_image`, `upscale_image`, `export_image` |
| changes a setting / adds data | `set_default_model`, `add_model` (the only tool that reaches the internet: Hugging Face, and GitHub for the engine) |
| destructive | `cancel_job` (throws away the work done so far; the queue is shared with the app) |

### The job pattern (important)

A picture takes **minutes**, but MCP clients usually give a tool call about 60 seconds. So the slow tools do not block:

```
generate_image {prompt: "a red fox in snow, golden hour"}        ->  job 3f9c1a2b7e40 queued, estimated 4.0 min
wait_for_job   {job_id: "3f9c1a2b7e40"}                          ->  Still running: denoising 3/8, 51 s elapsed, ~3 min left   (returns after <= 30 s)
wait_for_job   {job_id: "3f9c1a2b7e40"}                          ->  ...
wait_for_job   {job_id: "3f9c1a2b7e40"}                          ->  done: /Users/you/Pictures/ImageLoom/20261016-101500_img_a-red-fox-in-snow_482913.png  + a preview
```

- `generate_image`, `edit_image` and `upscale_image` return immediately with `job_id`, the model, the size and an **estimate** of the duration (learned from real timings on your machine).
- `wait_for_job` waits up to `timeout_seconds` (default 30, maximum 50) and returns early when the job ends. Call it again until `status` is `done` (or `error`).
  If the client sent a `progressToken`, it also emits `notifications/progress` (percent with a message such as `denoising 3/8, 51 s elapsed, ~3 min left`).
- `wait_seconds` (0-50) on the submitting tools is a shortcut for "submit, then wait that long once". Use it only for small jobs.
- One job runs at a time (a 16 GB machine cannot run two); more jobs queue. The queue is shared with the ImageLoom app and the CLI.
- `upscale_to` on `generate_image` is reported as **one** job: it stays `running` until the upscaled picture exists; the first-stage picture is listed as `originals`.
- Jobs live in the ImageLoom server's memory. After a restart `get_job` no longer knows them; `recent_images` still lists the finished pictures.
- Cancelling a `wait_for_job` call (the client's stop button) stops the waiting, not the job. Use `cancel_job` for that.

### generate_image

| Argument | Type | Notes |
|---|---|---|
| `prompt` | string, required | Up to 4000 characters. Describe a scene; put text to be drawn in single quotes. |
| `width`, `height` | integers | Together; multiples of 32 between 512 and 2752; `width*height` at most 2048x2048. |
| `aspect_ratio` | string | `1:1` (default), `4:3`, `16:9`, `9:16` ... Alternative to width/height; the pixel count comes from `quality`. |
| `quality` | `fast` (default) / `standard` / `hq` / `max` | Same as the CLI's `-q`: ~768 px (~4 min), ~1024 px (~7 min), ~1536 px (~30 min), ~2048 px (hours). |
| `model` | string | Id or part of the name (see `list_models`); default: the model chosen in the app. |
| `seed` | integer | For reproducible results. |
| `count` | 1-8 | Variants from one model load (each extra costs ~30 s less than the first). |
| `transparent` | boolean | Real alpha channel (models with the `alpha` ability, e.g. Qwen-Image 2.1). |
| `upscale_to` / `upscale_model` | `2x` `4x` `2k` `4k` `8k` / `quality` `fast` `illustration` | Chain an AI upscale after rendering. |
| `wait_seconds` | 0-50 | Optionally wait inline. |

### edit_image

`image_path` and `instruction` are required; optional `extra_references` (at most 2 more pictures the instruction can refer to), `quality`, `model`, `seed`, `wait_seconds`.
Editing is much slower than creating, so its tiers are smaller: `fast` ~512 px (~2.5 min), `standard` ~640 px (~8 min), `hq` ~768 px (~20 min), `max` ~1024 px. Needs an edit-capable model
(`edit` in `list_models`; the default Qwen-Image 2.1 is). The original is never modified.

### upscale_image

`image_path` and `target` (`2x` `4x` `2k` `4k` `8k`) are required; optional `model` (`quality` default, `fast` = drafts only, `illustration`) and `wait_seconds`.
The long side is limited to 8192 px. A PNG already in the ImageLoom folder is used in place; any other picture is converted to PNG and copied there first.

### export_image

`image_path` and `format` (`png` `webp` `jpeg` `svg`) are required; optional `quality` (50-100, default 90), `lossless` (WebP), `preset` (SVG: `logo` default, `illustration`, `detailed`, `bw`), `max_side`.
Synchronous; returns the new file (in `_exports/` of the ImageLoom folder). SVG is a tracing of the raster: good for logos and flat art, huge for photos.

### get_job

`job_id` (the id returned by a submitting tool). Returns `status` (`queued`, `running`, `done`, `error`, `cancelled`), `phase`, `step`/`steps`, `elapsed_seconds`, `estimated_seconds`, `remaining_seconds`
(an estimate), and for finished jobs the `images` (absolute `path`, `width`, `height`, `seed`) plus previews. `structuredContent` carries the same data as JSON.

### wait_for_job

`job_id`, `timeout_seconds` (1-50, default 30). Same result as `get_job`; if the time runs out first, the status is returned with an instruction to call again.

### cancel_job

`job_id`. Cancels a queued or running job (and the automatic upscale chained after it). Finished jobs are left alone.

### list_jobs

No arguments. The last 25 jobs the server knows, newest first.

### list_models

No arguments. `id`, `name`, `status` (`ready`, `downloading` with a progress fraction, `error`, `files_missing`), `size_bytes`, `edit`, `alpha`, `experimental`, `default`, and the free disk space.

### add_model

| Argument | Notes |
|---|---|
| `url` | `https://huggingface.co/owner/repo` or `owner/repo`. |
| `file` | Weights file from the plan's list (default: the best one that fits a 16 GB Mac). |
| `family`, `variant` | Override the detected family (`qwen-image-2.1`, `z-image`, `flux1`, `sdxl`, `sd15`, `custom`) or variant (`turbo`, `base`, `schnell`, `dev`). |
| `dry_run` | **`true` by default**: nothing is downloaded, you get the plan with sizes, companions, free disk space, and warnings. `false` starts the download in the background. |

Downloads are **many gigabytes** (typically 4-20 GB). The tool description tells the model to show the plan and ask the user first. Follow a running download with `list_models`.
Only Qwen-Image 2.1 Turbo is verified by the author; other families are marked `experimental`. Gated repositories need a Hugging Face token that you add in the ImageLoom app.

### set_default_model

`id` (exact, from `list_models`). The same setting as the app's model picker.

### recent_images

`limit` (1-50, default 10) and `previews` (0-4, default 0: attach downscaled previews of the newest ones). Each item: `path`, `kind` (`gen`, `edit`, `upscale`, `composite`), `prompt`, size, `seed`, `model`, `created`.

## What a result looks like

- `content`: a text block that always contains everything needed (ids, absolute paths, next step), followed by `image` blocks (base64 JPEG, or PNG when transparent) for finished pictures:
  **previews, downscaled to at most 768 px, at most 4 per result**. The files on disk keep their full resolution.
- `structuredContent` (when the client negotiated protocol 2025-06-18): the same facts as JSON, e.g. `{"job_id": ..., "status": "done", "images": [{"path": ..., "width": ..., "height": ..., "seed": ...}], ...}`.
- `isError: true` for failures (bad arguments, missing file, engine or model not installed, a failed job), with a message that says what to do. Protocol errors are used only for protocol problems (unknown tool, malformed JSON-RPC).

Protocol versions: `2025-06-18`, `2025-03-26` and `2024-11-05` (the server answers with the client's version when it knows it, else with `2025-06-18`). It implements `initialize`, `ping`, `tools/list`, `tools/call`,
`notifications/initialized` and `notifications/cancelled`, and ignores other notifications.

## Safety notes

- **Files.** Tools write only into the ImageLoom folder (usually `~/Pictures/ImageLoom`: pictures, `_uploads/` for inputs, `_exports/`) through the ImageLoom server, which refuses any path outside it. There is no
  tool that deletes or overwrites a file, and none takes an output path. A picture you pass in is converted to PNG and *copied* into `_uploads/`; the original is never touched.
- **What may be read.** Input paths must be existing regular files with an image extension **and** an image header (PNG, JPEG, WebP, GIF, BMP, TIFF, HEIC), at most 64 MB. A text file renamed to `.png` is refused. Only generated pictures from ImageLoom's own folder are ever sent back as previews.
- **Downloads.** `add_model` downloads gigabytes. It defaults to `dry_run=true`, its description tells the model to ask you first, and your client's tool-approval prompt is a second guard: consider requiring approval for `add_model`, `cancel_job` and the slow tools.
- **Compute.** `quality: hq` / `max` can run for half an hour to many hours and use most of a 16 GB machine's memory. The descriptions tell the model to start with `fast`.
- **No telemetry.** Prompts and pictures never leave your machine. The MCP server talks only to `127.0.0.1`; the ImageLoom server talks to Hugging Face / GitHub only when you (or your agent, after your OK) add a model or the engine is downloaded.
- **Trust model.** Same as the CLI: whoever can run your MCP client can use your local ImageLoom. There is no content filter in this stack (see the README's *Privacy and responsible use*).
- **stdout.** Nothing but protocol messages is ever written to stdout (file descriptor 1 is re-pointed to stderr inside the process, so even a stray `print` or a child process cannot corrupt the stream).

## Troubleshooting

| Symptom | What to check |
|---|---|
| The client says the server failed or disconnected | Run the exact command in a terminal: it must print `imageloom-mcp: imageloom ... ready on stdio` on stderr and then wait. Use absolute paths and a Python 3.9+ (`python3 --version`). |
| `the sd-cli engine is not installed yet` / `no model is installed yet` | Open the ImageLoom app (or `imageloom ui`) and finish the setup screen. An agent can only help with `add_model`, after your OK. |
| The call times out in the client | Lower `timeout_seconds` / `wait_seconds` (clients commonly allow 60 s, the tools stay under 55 s), or raise the client's tool timeout. The job keeps running; ask for it again with `wait_for_job`. |
| `no job with id ...` | The ImageLoom server was restarted (jobs are in memory). `recent_images` shows what was finished. |
| No image previews | `pip install pillow`, or use macOS (the built-in `sips` is used). Without either, only PNGs up to 768 px are attached; the path is always in the text. |
| `converting pictures needs macOS sips or the Pillow package` | Same cause, for `edit_image` / `upscale_image` / `export_image` with a picture that is not already a PNG in the ImageLoom folder. |
| `cannot reach the ImageLoom server` | It was quit. Calling a tool again starts it (the jobs tools do not start one, because jobs live in its memory). Check `<data folder>/logs/server.log`. |
| It uses a different port or data folder than the app | The app's port is found through `server.json` in the data folder. Set `IMAGELOOM_HOME` / `IMAGELOOM_PORT` in the client's `env` to pin them. |
| Something unexpected | Set `IMAGELOOM_MCP_DEBUG=1` and read the client's log for this server. |

Try it by hand (a scripted session; replace the path):

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"hand","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_models","arguments":{}}}' \
  | python3 /ABS/PATH/TO/imageloom/core/imageloom_mcp.py
```

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) (`npx @modelcontextprotocol/inspector python3 /ABS/PATH/TO/imageloom/core/imageloom_mcp.py`) is a convenient way to click through the tools.

## Limitations

- stdio only (no HTTP / SSE transport), no authentication. Resources and prompts are not offered, only tools.
- Written against the MCP specification revisions 2024-11-05, 2025-03-26 and 2025-06-18 and tested with scripted clients over real pipes (`tests/test_mcp.py`); it was **not** tried against the real Claude Desktop, Claude Code, Cursor, VS Code, Zed or Cline,
  so a client-specific quirk is possible: please open an issue with the client's log.
- Tool results can carry images, but not every client shows them to the model; the absolute paths are always in the text.
- One job at a time, minutes per image: that is the hardware, not the protocol.
