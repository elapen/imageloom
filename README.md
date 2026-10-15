<div align="center">

<img src="assets/icon_1024.png" width="112" alt="ImageLoom icon">

# ImageLoom

**A local AI image studio for Mac and Windows.**<br>
Chat with your own image model: create, edit with a brush, upscale to 4K, export transparent PNG / WebP / SVG.<br>
No cloud, no account, no Python setup, nothing leaves your computer.

[![Release](https://img.shields.io/github/v/release/elapen/imageloom?color=4a5cff)](https://github.com/elapen/imageloom/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-macOS%20(Apple%20Silicon)%20%C2%B7%20Windows%20(experimental)-lightgrey)](#download)
[![Engine](https://img.shields.io/badge/engine-stable--diffusion.cpp-orange)](https://github.com/leejet/stable-diffusion.cpp)
[![Русский](https://img.shields.io/badge/README-%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue)](README.ru.md)

<img src="docs/images/demo.gif" width="860" alt="ImageLoom tour: chat, model and effort pickers, add a model from a link, gallery">

</div>

> Every picture on this page was generated on a laptop with ImageLoom. No API keys, no subscription.

## Why ImageLoom

- **It feels like a chat, not a node graph.** Separate chats, each remembering its own model and "effort"; the result of one message is the starting photo of the next ("make it night").
- **One app for the whole job.** Create → edit by instruction or with a **brush mask** → **upscale ×2 / ×4 / 2K / 4K / 8K** → export **PNG, WebP, JPEG, SVG** with a real transparent background.
- **New model on release day.** Paste a Hugging Face link, pick a file, it downloads what the model needs and shows up in the picker. No YAML, no scripts.
- **Light and private.** A native window (not a browser tab), ~160 MB without models, one local server that starts with the app and stops, together with any running job, when you quit.
- **Honest about speed.** Estimates come from timings measured on your machine and are labelled "estimate" until a model has been timed.
- **Built for people and for agents.** A tiny CLI (`imageloom "a red fox in snow" --upscale 4k`) and ready-made skills for Claude Code, Cursor and Hermes.

## Made with ImageLoom

<table>
<tr>
<td><img src="docs/images/g-cabin.jpg" alt="Cabin in a snowy forest, Z-Image Turbo"></td>
<td><img src="docs/images/g-poster.jpg" alt="Swiss Alps poster with lettering, Qwen-Image"></td>
<td><img src="docs/images/g-watercolor.jpg" alt="Watercolor lighthouse, Z-Image Turbo"></td>
</tr>
<tr>
<td><img src="docs/images/g-isometric.jpg" alt="Isometric home office, Z-Image Turbo"></td>
<td><img src="docs/images/g-sneakers.jpg" alt="Studio product shot, Z-Image Turbo"></td>
<td><img src="docs/images/g-fox.jpg" alt="Fox logo with transparent background, Qwen-Image"></td>
</tr>
</table>

<sub>Cabin 1024×576 in 166 s, the others 512-768 px in 1.5-5 minutes on a base M4 (16 GB). The fox logo is a real transparent PNG. Prompts are in the app's history.</sub>

**Edit by describing the change** (Qwen-Image, 512 px, 163 s):

<img src="docs/images/edit-before-after.jpg" width="760" alt="Before and after: turn the scene into a snowy night with northern lights">

## Screenshots

| | |
|---|---|
| <img src="docs/images/showcase-chat.png" alt="Chat with model and effort pickers"> | <img src="docs/images/showcase-edit.png" alt="Edit with a brush mask"> |
| **Pick model and effort per chat** like in ChatGPT or Claude | **Brush-mask editing** pastes only the edited pixels into your full-resolution photo |
| <img src="docs/images/showcase-addmodel.png" alt="Add a model from a Hugging Face link"> | <img src="docs/images/showcase-gallery.png" alt="Gallery"> |
| **Add a model by link**: family, file, size and free space are checked first | **Gallery** with versions, before/after slider, favourites, 4K upscale |

<details><summary>A whole conversation, and the Russian UI</summary>

<img src="docs/images/hero.png" width="640" alt="A conversation: create, then edit the result"> <img src="docs/images/hero-ru.png" width="640" alt="Русский интерфейс">

</details>

## Download

| Platform | Get it | Notes |
|---|---|---|
| **macOS** (Apple Silicon, macOS 13+) | [`ImageLoom-1.0.0-macOS-arm64.dmg`](https://github.com/elapen/imageloom/releases/latest) | Drag to Applications. The app is not notarized: right-click → **Open** the first time (or System Settings → Privacy & Security → **Open Anyway**). |
| macOS via Homebrew | `brew install --cask elapen/tap/imageloom` | Same app, installed by cask. |
| **Windows** 10/11 x64, *experimental* | [`ImageLoom-1.0.0-Windows-x64-Setup.exe`](https://github.com/elapen/imageloom/releases/latest) or the portable `.zip` | Built from macOS and **not yet run on real Windows**: please report what you see. SmartScreen will warn (unsigned). Needs a Vulkan GPU. |
| From source (any OS with Python 3.9+) | `pip install -r core/requirements.txt && python3 core/server.py` | Open http://127.0.0.1:7870. Linux is untested. |

**First run.** A wizard lists exactly what it will download (engine, the default model, upscalers) and how big it is before it starts, and shows progress and speed. The default
**Qwen-Image 2.1 Turbo Q8** is about 18 GB; **Z-Image Turbo Q4** is a 4 GB alternative you can add from the model picker. After the downloads the app works fully offline.

**Hardware.** Apple Silicon with **16 GB RAM** is the sweet spot (Qwen-Image peaks at ~9 GB while creating, ~10.5 GB while editing). More RAM and a faster chip simply make it faster.
Windows needs a GPU with Vulkan support.

## What it can do

- **Create** from a prompt (English, Russian and other languages the model understands), with aspect ratio presets, 1–8 images per run, seeds, and style/quick chips.
- **Edit** a photo by describing the change, or paint a **mask** and describe what should happen inside it. Add extra reference photos for "put the person from image 2 into image 1".
- **Upscale** with Real-ESRGAN models (quality / fast / illustration), or chain it automatically after every render: draft at 512 px, 4K a minute later.
- **Export**: PNG, WebP (lossy or lossless), JPEG, and **SVG** (vector tracing with logo / illustration / detailed / black-and-white presets). Ask for a transparent background and get real alpha.
- **Organize**: chats, gallery, favourites, version tree for each image, before/after slider, "reveal in Finder / Explorer", move to Trash.
- **Notify**: a desktop notification when a long render finishes. Quitting the app while a job runs asks first.
- **Languages**: the UI is available in English and Russian (auto-detected, switchable).
- **Shortcuts**: `⌘K` command palette, `⌘N` new chat, `/` focus the prompt, `?` help.

## Models

| Model | Create | Edit | Transparent | Status |
|---|---|---|---|---|
| **Qwen-Image 2.1 Turbo** (default, Q8) | yes | yes | yes | verified on Apple M4 16 GB |
| **Z-Image Turbo** (Q4_K) | yes | no | no | verified on Apple M4 16 GB |
| FLUX.1 · SDXL · SD 1.5 | yes | no | no | recipes from the sd.cpp docs, **experimental** (not run by the author) |

Details, warnings and how to add a family: [docs/MODELS.md](docs/MODELS.md).

## Speed (base M4, 16 GB, one image)

| | 512² | 768² | 1024² |
|---|---|---|---|
| Qwen-Image 2.1 Turbo Q8, create | 110 s | 239 s | 395 s |
| Z-Image Turbo Q4_K, create | 84 s | 151 s | ~250 s (est.) |
| Qwen-Image edit | 152 s | 20 min (swaps) | not worth it |
| Upscale 1024² → 4096², Real-ESRGAN quality | | | 245 s |

This is a laptop, not an RTX card: diffusion on 16 GB of unified memory is slow but steady. The best workflow is a small draft first, then upscale the keeper ([benchmarks](docs/BENCHMARKS.md)).

## Command line and AI agents

```bash
imageloom "vintage travel poster of the Swiss Alps, lettering 'SWISS ALPS'" -s 768x1024
imageloom edit photo.jpg "replace the sky with a dramatic sunset" -o edited.png
imageloom "a red fox head logo, flat vector style" --transparent -o fox.png
imageloom export fox.png --format svg --preset logo
imageloom upscale picture.png --to 4k
imageloom models add https://huggingface.co/leejet/Z-Image-Turbo-GGUF --dry-run
imageloom batch prompts.txt  # one JSON result per non-empty line
```

Batch files accept one prompt per line. Add optional `size=WxH` and `model=ID` fields anywhere on a line; quoted prompt text is also accepted. Blank lines and lines beginning with `#` are ignored.

The CLI needs only Python 3.9+. It talks to the server of a running app, or starts one. Copy [`skills/imageloom`](skills/imageloom/SKILL.md) into `~/.claude/skills/` or
`~/.cursor/skills/` (a Hermes variant is in [`skills/hermes`](skills/hermes/imageloom/SKILL.md)) and your coding agent can create and edit images for your projects on its own.

## Privacy and responsible use

- Prompts, images and history never leave your computer. The only network traffic is what you trigger: downloads from Hugging Face and the engine release from GitHub. There is no telemetry.
- The server listens on `127.0.0.1` only and rejects cross-site requests. A Hugging Face token (for gated repos) is optional and stored locally.
- **There is no content filter in this stack.** The default model's text encoder is a Qwen3-VL variant with its refusal behaviour removed, and diffusion models have no filter of their own. You are responsible for
  what you create and share: do not make illegal content, sexual content involving minors, or realistic images of real people without their consent. Model licenses apply to the results
  (the default Qwen weights are research / non-commercial); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## FAQ

**Why is it slow?** Diffusion on 16 GB of shared memory is slow but steady (see the table above). Use *Fast* effort or Z-Image for drafts and upscale the keeper.
**Can I use my own `sd-cli`?** Yes: `IMAGELOOM_SD_CLI=/path/to/sd-cli`.
**Where are my files?** Images in `~/Pictures/ImageLoom`, models and settings in `~/Library/Application Support/ImageLoom` (Windows: `%LOCALAPPDATA%\ImageLoom`). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
**Why not ComfyUI / Draw Things / Fooocus?** They are great and more powerful in places. ImageLoom aims at a smaller, chat-first workflow with one-click upscale/export and an agent-friendly CLI.
Use whatever fits your hands.
**Is it signed?** Not yet; see the download notes. Checksums are in each release.

## Contributing

Bug reports, model recipes, translations, Windows and Linux testing are all welcome: [CONTRIBUTING.md](CONTRIBUTING.md). Ideas and questions go to
[Discussions](https://github.com/elapen/imageloom/discussions). If ImageLoom is useful to you, a ⭐ helps others find it.

## Credits

Built on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) by leejet, the Qwen-Image and Z-Image teams, AtomicChat's GGUF conversions,
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [vtracer](https://github.com/visioncortex/vtracer), and python-build-standalone. Full list: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Code: [MIT](LICENSE).
