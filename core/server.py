#!/usr/bin/env python3
"""ImageLoom core server: queue + HTTP API + Studio UI around stable-diffusion.cpp (sd-cli).

Engine: Qwen-Image 2.1 Turbo, Q8_0 denoiser (AtomicChat GGUF) + abliterated Q8_0 Qwen3-VL text
encoder (AtomicChat) + Comfy-Org VAE, run by the official stable-diffusion.cpp `sd-cli` (Metal).
Modes: `generate` (text to image) and `edit` (1-3 reference images + an instruction).
Each job is one `sd-cli` subprocess, one job at a time: on a 16 GB Mac the subprocess hands every
byte back to the OS between jobs, and the server itself stays tiny. Pure stdlib, 127.0.0.1 only.
"""
from __future__ import annotations

import atexit
import errno
import glob
import hashlib
import http.client
import io
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import socketserver
import ssl
import struct
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M  # noqa: E402  (registry, family recipes, Hugging Face inspection)

# Outgoing connections try IPv4 first. Python (unlike curl) walks the address list in order without "happy eyeballs",
# so on a network whose IPv6 is not routable every request to a dual-stack host (Hugging Face, GitHub) stalls for
# minutes. IPv6 stays as the fallback.
_real_getaddrinfo = socket.getaddrinfo


def _ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
    res = _real_getaddrinfo(host, port, family, type, proto, flags)
    return sorted(res, key=lambda r: 0 if r[0] == socket.AF_INET else 1)


socket.getaddrinfo = _ipv4_first

SYSTEM = platform.system()  # Darwin | Windows | Linux
HOME = Path.home()
CODE = Path(__file__).resolve().parent  # server.py and web/ live here (read-only inside an installed app)


def default_data_dir() -> Path:
    """Where models, history, thumbnails and settings live (never inside the app bundle)."""
    env = os.environ.get("IMAGELOOM_HOME")
    if env:
        return Path(os.path.abspath(os.path.expanduser(env)))  # absolute: paths handed to sd-cli can never look like options
    if SYSTEM == "Darwin":
        return HOME / "Library" / "Application Support" / "ImageLoom"
    if SYSTEM == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", str(HOME / "AppData" / "Local"))) / "ImageLoom"
    return Path(os.environ.get("XDG_DATA_HOME", str(HOME / ".local" / "share"))) / "imageloom"


def die(msg: str):
    sys.exit(f"imageloom: {msg}")


def make_dir(path: Path, what: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        die(f"cannot create the {what} folder {path}: {exc.strerror or exc}\n  (point IMAGELOOM_HOME, or output_dir in config.json, at a writable folder)")


DATA = default_data_dir()
make_dir(DATA, "data")
EXE = ".exe" if SYSTEM == "Windows" else ""


def pictures_dir() -> Path:
    """The real Pictures folder (OneDrive can relocate it on Windows)."""
    if SYSTEM == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as k:
                return Path(os.path.expandvars(winreg.QueryValueEx(k, "My Pictures")[0]))
        except Exception:  # noqa: BLE001
            pass
    return HOME / "Pictures"


def find_sd_cli() -> Path:
    """Engine lookup: explicit env, user data, bundled with the app, then PATH."""
    cands = []
    if os.environ.get("IMAGELOOM_SD_CLI"):
        cands.append(Path(os.environ["IMAGELOOM_SD_CLI"]))
    if os.environ.get("IMAGELOOM_ENGINE_DIR"):
        cands.append(Path(os.environ["IMAGELOOM_ENGINE_DIR"]) / f"sd-cli{EXE}")
    cands += [DATA / "sd" / "bin" / f"sd-cli{EXE}", DATA / "engine" / f"sd-cli{EXE}", CODE.parent / "engine" / f"sd-cli{EXE}", CODE.parent.parent / "engine" / f"sd-cli{EXE}", CODE / "engine" / f"sd-cli{EXE}"]
    for c in cands:
        if c.is_file():
            return c
    found = shutil.which("sd-cli")
    return Path(found) if found else cands[1]  # (not found: the setup wizard offers to download it)


CONFIG_PATH = DATA / "config.json"
MODELS = DATA / "models" / "qwen21-turbo-q8-uncensored"
DEFAULTS = {
    "port": 7870,
    "diffusion": str(MODELS / "Qwen-Image-2.1-Turbo-Q8_0.gguf"),
    "llm": str(MODELS / "Qwen-Image-2.1-Turbo-Abliterated-Uncensored-Q8_0.gguf"),
    "mmproj": str(MODELS / "mmproj-Qwen-Image-2.1-Turbo-Abliterated-Uncensored-F16.gguf"),
    "vae": str(MODELS / "vae" / "qwen_image_2.1_vae_bf16.safetensors"),
    "min_bytes": {"diffusion": 7.5e9, "llm": 8.6e9, "vae": 0.6e9},
    "min_bytes_mmproj": 1.1e9,
    # Turbo schedule from the checkpoint's model_index.json (+ final 0.0), required for the intended look
    "sigmas": "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0",
    "steps": 8,
    # Metal (macOS): VAE on the GPU; elsewhere sd-cli's auto-fit places the weights
    "extra_args": ["--rng", "cpu", "--diffusion-fa", "--mmap", "--backend", "vae=MTL0", "--vae-tiling"] if SYSTEM == "Darwin" else ["--rng", "cpu", "--diffusion-fa", "--mmap", "--vae-tiling"],
    "output_dir": str(pictures_dir() / "ImageLoom"),
    # measured end to end on a base M4 / 16 GB: [pixels, seconds for one image]
    "eta_points": [[262144, 110], [589824, 239], [1048576, 395], [2359296, 1882]],
    # editing carries the reference image's tokens too (sequence length doubles): measured: 512x512 152 s, 768x768 1208 s
    "edit_points": [[262144, 152], [589824, 1208]],
    "extra_image_discount": 30,  # each further image in one run skips the encoder load
    "max_pixels": 2048 * 2048,
    "warn_pixels": 1100000,  # above this the UI asks for confirmation (long, heavy on 16 GB)
    "max_pixels_edit": 1024 * 1024,
    "warn_pixels_edit": 420000,  # editing above ~640x640 takes very long on 16 GB
    "min_side": 512,
    "max_side": 2752,
    "multiple": 32,  # Qwen-Image 2.1 needs dimensions divisible by 32
    # ---- upscaling (sd-cli -M upscale with ESRGAN-family weights) ----
    "upscale_dir": str(DATA / "models" / "upscale"),
    # seconds per INPUT pixel for one 4x pass, measured on 1024x1024 -> 4096x4096: Nomos2 38 s, x4plus 245 s
    "upscale_rate": {"fast": 3.6e-5, "quality": 2.34e-4, "illustration": 1.0e-4},
    "upscale_overhead": 8,
    "max_upscale_side": 8192,
}


def load_config() -> dict:
    """DEFAULTS overridden by config.json. Unknown keys and values of the wrong type are ignored with a warning."""
    cfg = dict(DEFAULTS)
    for k, v in M.read_json(CONFIG_PATH, {}).items():
        want = DEFAULTS.get(k)
        num = isinstance(want, (int, float)) and not isinstance(want, bool)
        if k in DEFAULTS and (isinstance(v, (int, float)) and not isinstance(v, bool) if num else isinstance(v, type(want))):
            cfg[k] = v
        else:
            M.warn(f"config.json: ignoring {k!r} ({'unknown setting' if k not in DEFAULTS else 'wrong type'})")
    if not 0 <= cfg["port"] < 65536:
        M.warn("config.json: 'port' must be 0..65535, using the default")
        cfg["port"] = DEFAULTS["port"]
    if not all(isinstance(a, str) for a in cfg["extra_args"]):
        M.warn("config.json: 'extra_args' must be a list of strings, using the default")
        cfg["extra_args"] = DEFAULTS["extra_args"]
    return cfg


CFG = load_config()
CFG["sd_cli"] = str(find_sd_cli())  # resolved again on every use, so a freshly downloaded engine is picked up


def builtin_model() -> dict:
    """The model the app ships its measurements for (Qwen-Image 2.1 Turbo Q8 with the encoder without refusals)."""
    return {"id": "qwen21-turbo-q8", "name": "Qwen-Image 2.1 Turbo · Q8", "family": "qwen-image-2.1", "variant": "turbo", "builtin": True, "verified": True,
            "files": {"diffusion": CFG["diffusion"], "llm": CFG["llm"], "vae": CFG["vae"], "llm_vision": CFG["mmproj"]}, "status": "ready",
            "tags": ["no refusals", "editing", "transparent background", "Russian"], "license": "Qwen Research (non-commercial)", "size": 18_100_000_000,
            "source": "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF",
            "notes": "Main model: Q8, text encoder without refusals, instruction-based editing, transparent background, Russian language. Timings were measured on this Mac."}


REG = M.Registry(DATA / "models.json", builtin_model(), DATA)
ETA_HINT = {"qwen-image-2.1": 1.0, "z-image": 0.55, "flux1": 1.1, "sdxl": 0.6, "sd15": 0.2, "custom": 1.0}  # first guess until real runs are recorded
STATS_PATH = DATA / "eta_stats.json"
CHATS_PATH = DATA / "chats.json"
OUT = Path(os.path.abspath(os.path.expanduser(CFG["output_dir"])))
UPLOADS = OUT / "_uploads"
THUMBS = DATA / "thumbs"
make_dir(OUT, "output")
make_dir(UPLOADS, "output")
make_dir(THUMBS, "thumbnail")
HISTORY = DATA / "history.jsonl"
WEB = CODE / "web"
VERSION = "1.0.0"
UPSCALE_MODELS = {  # key -> (file, UI label, note)
    "quality": ("RealESRGAN_x4plus.pth", "Quality (recommended)", "Real-ESRGAN x4plus: clean and even, no pattern; ~4 min per megapixel of input"),
    "fast": ("4xNomos2_hq_mosr.pth", "Fast (draft)", "Nomos2 MoSR: 6x faster (~40 s per megapixel), but leaves a visible weave pattern on smooth areas (sky, background, foam). For checking, not for the final result"),
    "illustration": ("RealESRGAN_x4plus_anime_6B.pth", "Illustration", "for posters, drawings and anime"),
}
UPSCALE_TARGETS = {"2x": ("factor", 2), "4x": ("factor", 4), "2k": ("side", 2560), "4k": ("side", 3840), "8k": ("side", 7680)}

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_RE = re.compile(r"\|[=>\s]*\|\s*(\d+)/(\d+)\s*-\s*([\d.]+)\s*(?:s/it|it/s)")
IMG_RE = re.compile(r"generating image:\s*(\d+)/(\d+)\s*-\s*seed\s*(-?\d+)")
PEAK_RE = re.compile(r"(\d+)\s+peak memory footprint")
PNG_SIG = b"\x89PNG\r\n\x1a\n"
IMG_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".svg": "image/svg+xml"}  # what /img may serve
MAX_UPLOAD = 64 * 1024 * 1024  # largest request body (an uploaded picture)
MAX_JSON = 1_000_000  # largest JSON request body
MAX_IMAGE_PIXELS = 100_000_000  # an upload with more pixels is refused (a decoded 100 MP RGBA picture is 400 MB)
MAX_IMAGE_SIDE = 16384
MAX_PROMPT = 4000  # also keeps the sd-cli command line far below the Windows 32 K limit
MAX_QUEUED = 50
MAX_JOBS_KEPT = 200  # finished jobs remembered for /api/jobs
MAX_HISTORY = 50_000
MAX_CHATS = 500
MAX_CONNECTIONS = 64  # simultaneous requests; stalled clients are also cut off by Handler.timeout
MAX_MODELS = 100
MAX_FILE_BYTES = 80 * 10**9  # sanity bound for one model file

JOBS: dict[str, dict] = {}
JOB_ORDER: list[str] = []
QUEUE: list[str] = []
HIST: list[dict] = []
LOCK = threading.Lock()
WAKE = threading.Event()
CURRENT: dict = {"id": None, "proc": None}


# ----------------------------------------------------------------------------------------- platform layer
try:
    from PIL import Image  # bundled with the app; the macOS fallback below uses `sips`
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS  # Pillow's own guard (it only errors beyond twice this); open_image() enforces the limit exactly
except Exception:  # noqa: BLE001
    Image = None


def open_image(path: Path):
    """Image.open + the pixel limit, checked from the header before a single pixel is decoded (decompression bombs)."""
    im = Image.open(path)
    if im.width * im.height > MAX_IMAGE_PIXELS:
        im.close()
        raise ValueError(f"picture is too large to process (more than {MAX_IMAGE_PIXELS // 10**6} megapixels)")
    return im
THUMB_SLOTS = threading.Semaphore(2)  # decoding a big picture costs hundreds of MB: at most two thumbnails at a time


def spawn(cmd: list[str]) -> subprocess.Popen:
    """Start sd-cli in its own process group (so a cancel kills the whole tree)."""
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if SYSTEM == "Windows" else {"start_new_session": True}
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, **kw)


def kill_tree(proc, force: bool = False) -> None:
    try:
        if SYSTEM == "Windows":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL if force else signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def time_prefix() -> list[str]:
    """macOS only: `/usr/bin/time -l` reports the peak memory footprint of a job."""
    return ["/usr/bin/time", "-l"] if SYSTEM == "Darwin" and Path("/usr/bin/time").exists() else []


def resize_long_side(src: Path, dst: Path, long_side: int) -> bool:
    if Image is not None:
        with open_image(src) as im:
            k = long_side / max(im.size)
            im.convert("RGBA" if im.mode in ("RGBA", "LA") else "RGB").resize((round(im.width * k), round(im.height * k)), Image.LANCZOS).save(dst, "PNG", compress_level=4)
        return True
    if SYSTEM == "Darwin":
        subprocess.run(["sips", "-Z", str(long_side), str(src), "--out", str(dst)], capture_output=True, check=False)
        return dst.exists()
    return False


def make_thumb(src: Path, dst: Path, size: int) -> bool:
    if Image is not None:
        with open_image(src) as im:
            keep_alpha = im.mode in ("RGBA", "LA") or "transparency" in im.info
            if not keep_alpha:
                im.draft("RGB", (size * 2, size * 2))
            im = im.convert("RGBA" if keep_alpha else "RGB")
            im.thumbnail((size, size), Image.LANCZOS)
            if dst.suffix == ".webp":
                im.save(dst, "WEBP", quality=80, method=4)
            else:
                im.convert("RGB").save(dst, "JPEG", quality=82, optimize=True)
        return True
    if SYSTEM == "Darwin":
        subprocess.run(["sips", "-Z", str(size), "-s", "format", "jpeg", "-s", "formatOptions", "82", str(src), "--out", str(dst)], capture_output=True, check=False)
        return dst.exists()
    return False


def move_to_trash(path: Path) -> None:
    """Recoverable delete: the OS trash / recycle bin; never an irreversible unlink."""
    try:
        from send2trash import send2trash
        send2trash(str(path))
        return
    except Exception:  # noqa: BLE001
        pass
    if SYSTEM == "Darwin" and Path("/usr/bin/trash").exists():
        subprocess.run(["/usr/bin/trash", str(path)], check=False)
        return
    bin_dir = DATA / "trash"
    bin_dir.mkdir(exist_ok=True)
    shutil.move(str(path), str(bin_dir / f"{int(time.time())}_{path.name}"))


def reveal(path: Path) -> None:
    if SYSTEM == "Darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
    elif SYSTEM == "Windows":
        subprocess.run(["explorer", "/select,", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path.parent)], check=False)


# ----------------------------------------------------------------------------------------- helpers
def sd_cli() -> str:
    return str(find_sd_cli())


def models_ready() -> bool:
    """The built-in default model is complete (what the first-run wizard installs)."""
    if not Path(sd_cli()).is_file():
        return False
    for key, need in CFG["min_bytes"].items():
        p = Path(CFG[key])
        if not p.is_file() or p.stat().st_size < need:
            return False
    return True


def model_ready(m: dict) -> bool:
    if m.get("builtin"):
        return models_ready()
    if not Path(sd_cli()).is_file() or m.get("status", "ready") != "ready":
        return False
    for role in M.FAMILIES[m["family"]]["required"]:
        p = (m.get("files") or {}).get(role)
        if not p or not Path(p).is_file() or Path(p).stat().st_size < 1_000_000:
            return False
    return True


def model_can_edit(m: dict) -> bool:
    caps = {**M.FAMILIES[m["family"]]["caps"], **(m.get("caps") or {})}
    p = (m.get("files") or {}).get("llm_vision")
    return bool(model_ready(m) and caps.get("edit") and p and Path(p).is_file() and Path(p).stat().st_size > 1_000_000)


def any_model_ready() -> bool:
    return any(model_ready(m) for m in REG.all())


def upscale_model_path(key: str) -> Path:
    return Path(CFG["upscale_dir"]) / UPSCALE_MODELS[key][0]


def upscale_available() -> dict:
    return {k: upscale_model_path(k).is_file() and upscale_model_path(k).stat().st_size > 5_000_000 for k in UPSCALE_MODELS}


def upscale_plan(src_w: int, src_h: int, target: str) -> dict:
    """Output size and the number of 4x ESRGAN passes needed for a target like 2x / 4x / 4k."""
    if target not in UPSCALE_TARGETS:
        raise ValueError("target must be one of " + ", ".join(UPSCALE_TARGETS))
    kind, val = UPSCALE_TARGETS[target]
    long_side = max(src_w, src_h)
    out_long = long_side * val if kind == "factor" else val
    if out_long > CFG["max_upscale_side"]:
        raise ValueError(f"result would be {out_long}px on the long side; the limit is {CFG['max_upscale_side']}")
    repeats = 2 if out_long > long_side * 4 * 2 else 1  # a second pass only when 4x is far from enough
    native = long_side * (16 if repeats == 2 else 4)
    k = out_long / long_side
    return {"out_w": round(src_w * k), "out_h": round(src_h * k), "out_long": out_long, "repeats": repeats, "native_long": native}


def eta_upscale(src_w: int, src_h: int, model: str, target: str) -> float:
    plan = upscale_plan(src_w, src_h, target)
    first = CFG["upscale_rate"][model] * src_w * src_h
    second = CFG["upscale_rate"][model] * (src_w * src_h * 16) if plan["repeats"] == 2 else 0
    return CFG["upscale_overhead"] + first + second


def edit_ready() -> bool:
    p = Path(CFG["mmproj"])
    return models_ready() and p.is_file() and p.stat().st_size >= CFG["min_bytes_mmproj"]


def _interp(pts: list, px: int) -> float:
    """Log-log interpolation over measured [pixels, seconds] points; extrapolates with at least a 1.15 exponent."""
    pts = sorted(pts)
    if px <= pts[0][0]:
        return pts[0][1] * (px / pts[0][0])
    for (p0, t0), (p1, t1) in zip(pts, pts[1:]):
        if px <= p1:
            return t0 * (px / p0) ** (math.log(t1 / t0) / math.log(p1 / p0))
    (p0, t0), (p1, t1) = pts[-2], pts[-1]
    return t1 * (px / p1) ** max(math.log(t1 / t0) / math.log(p1 / p0), 1.15)


def _load_stats() -> dict:
    return M.read_json(STATS_PATH, {})


STATS: dict = _load_stats()


def record_stat(model_id: str, kind: str, px: int, seconds: float) -> None:
    """Remember how long a single-image run really took, so the estimate for this model improves with use."""
    with LOCK:
        rows = STATS.setdefault(model_id, {}).setdefault(kind, [])
        rows.append([px, round(seconds, 1)])
        del rows[:-12]
        try:
            M.atomic_write(STATS_PATH, json.dumps(STATS), sync=False)
        except OSError:
            pass


def eta_info(w: int, h: int, n: int = 1, edit: bool = False, model: dict | None = None) -> tuple[float, bool]:
    """(seconds, is_guess). The built-in model uses measured curves; other models start from a family-based guess and
    are re-scaled by their own recorded runs."""
    m = model or REG.get(None)
    pts = CFG["edit_points" if edit else "eta_points"]
    guess = False
    if not m.get("builtin"):
        k = ETA_HINT.get(m["family"], 1.0) * m.get("eta_scale", 1.0)
        pts = [[p, s * k] for p, s in pts]
        learned = (STATS.get(m["id"]) or {}).get("edit" if edit else "gen", [])
        if learned:
            ratios = sorted(sec / _interp(pts, px) for px, sec in learned)
            pts = [[p, s * ratios[len(ratios) // 2]] for p, s in pts]
        else:
            guess = True
    one = _interp(pts, w * h)
    return one + (n - 1) * max(one - CFG["extra_image_discount"], 0), guess


def eta_seconds(w: int, h: int, n: int = 1, edit: bool = False, model: dict | None = None) -> float:
    return eta_info(w, h, n, edit, model)[0]


_TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя", ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya"]))


class ApiError(ValueError):
    """A request problem that carries its HTTP status (plain ValueError means 400)."""

    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


def clamp_int(v, lo: int, hi: int) -> int:
    try:
        return max(lo, min(int(v), hi))
    except (TypeError, ValueError, OverflowError):
        return lo


def as_int(v, name: str, lo: int, hi: int, default: int | None = None) -> int:
    """A whole number from JSON/query input, or a 400 that names the field (instead of a TypeError further down)."""
    if v is None or v == "":
        if default is None:
            raise ValueError(f"{name} is required")
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"{name} must be a number")
    try:
        n = int(v)
    except (ValueError, OverflowError):
        raise ValueError(f"{name} must be a whole number") from None
    if not lo <= n <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return n


def slugify(text: str) -> str:
    """ASCII file-name slug; Cyrillic is transliterated so Russian prompts still give readable names."""
    s = "".join(_TR.get(ch, ch) for ch in text.lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:40].strip("-") or "image"


def png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != PNG_SIG or data[12:16] != b"IHDR":
        raise ValueError("not a PNG file")
    return struct.unpack(">II", data[16:24])


def check_png(data: bytes) -> tuple[int, int]:
    """Uploads must be real PNGs of sane size: header, dimensions and (with Pillow) every chunk checksum."""
    w, h = png_size(data)
    if not (0 < w <= MAX_IMAGE_SIDE and 0 < h <= MAX_IMAGE_SIDE and w * h <= MAX_IMAGE_PIXELS):
        raise ValueError(f"picture is too large (at most {MAX_IMAGE_SIDE} px per side and {MAX_IMAGE_PIXELS // 10**6} megapixels)")
    if Image is not None:
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
        except Exception as exc:
            raise ValueError("not a valid PNG file") from exc
    return w, h


def read_png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as fh:
        return png_size(fh.read(64))


def png_has_alpha(path: Path) -> bool:
    """True when the picture really has transparent pixels. sd-cli always writes RGBA PNGs, so the header alone proves nothing:
    check the alpha channel's minimum (a Pillow decode, about 50 ms per megapixel)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(33)
        if head[:8] != PNG_SIG or head[25] not in (4, 6):
            return False
        if Image is None:
            return False
        with open_image(path) as im:
            if im.mode not in ("RGBA", "LA"):
                return False
            lo, _hi = im.getchannel("A").getextrema()
            return lo < 250
    except Exception:  # noqa: BLE001
        return False


def rel_ok(rel) -> bool:
    """Syntax check of a library file name: relative, forward slashes, no drive, NUL or control characters."""
    return (isinstance(rel, str) and 0 < len(rel) <= 400 and not rel.startswith("/") and not re.search(r"[\\:\x00-\x1f]", rel)
            and all(seg not in ("", ".", "..") for seg in rel.split("/")))


def safe_path(rel: str) -> Path:
    """A picture inside the output folder. Absolute paths, `..`, backslashes, drive letters, NUL bytes, non-image extensions
    and symlinks leading outside are all refused."""
    if not rel_ok(rel):
        raise ValueError("bad file name")
    target = (OUT / rel).resolve()
    if OUT.resolve() not in target.parents or target.suffix.lower() not in IMG_TYPES:
        raise ValueError("path outside the output folder")
    return target


# ----------------------------------------------------------------------------------------- history
def load_history() -> None:
    """Read history.jsonl. Unreadable lines are dropped (the original is kept as history.jsonl.bad-<time>) instead of crashing."""
    try:
        text = HISTORY.read_text(encoding="utf-8-sig", errors="replace")
    except FileNotFoundError:
        return
    except OSError as exc:
        return M.warn(f"cannot read {HISTORY}: {exc.strerror or exc}")
    rows, bad = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(row, dict) and isinstance(row.get("file"), str):
            row["file"] = row["file"].replace("\\", "/")  # rows written on Windows by older builds
        if not isinstance(row, dict) or not rel_ok(row.get("file")):
            bad += 1
            continue
        row.setdefault("id", uuid.uuid4().hex[:10])
        row.setdefault("kind", "gen")
        row.setdefault("parent", None)
        row.setdefault("refs", [])
        row.setdefault("fav", False)
        rows.append(row)
    if bad:
        M.quarantine(HISTORY, f"{bad} unreadable lines skipped", copy_only=True)
    HIST[:] = rows[-MAX_HISTORY:]
    if bad or len(rows) > MAX_HISTORY or (text and not text.endswith("\n")):  # also seals a half-written last line
        with LOCK:
            persist_history()


def backfill_alpha() -> None:
    """Older history rows have no `alpha` flag: fill it in quietly after startup, then save once."""
    changed = False
    for r in HIST:
        if "alpha" not in r:
            r["alpha"] = png_has_alpha(OUT / r["file"])
            changed = True
    if changed:
        with LOCK:
            persist_history()


def persist_history() -> None:
    M.atomic_write(HISTORY, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in HIST))


def add_rows(rows: list[dict]) -> None:
    for r in rows:
        r.setdefault("alpha", png_has_alpha(OUT / r["file"]))
    with LOCK:
        HIST.extend(rows)
        if len(HIST) > MAX_HISTORY:
            del HIST[:len(HIST) - MAX_HISTORY]
            persist_history()
        else:
            with HISTORY.open("a", encoding="utf-8") as fh:
                fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def visible_history() -> list[dict]:
    with LOCK:
        snapshot = list(HIST)
    rows = [r for r in snapshot if (OUT / r["file"]).exists()]  # (the stat calls run outside the lock)
    return rows[::-1][:600]


# ----------------------------------------------------------------------------------------- jobs
def validate_upscale(body: dict) -> dict:
    model = str(body.get("model", "quality"))
    if model not in UPSCALE_MODELS:
        raise ValueError("model must be one of " + ", ".join(UPSCALE_MODELS))
    if not upscale_available()[model]:
        raise ValueError(f"upscaler weights for '{model}' are missing; run `imageloom doctor`")
    src = str(body.get("src", ""))
    path = safe_path(src)
    if not path.is_file():
        raise ValueError(f"source image not found: {src}")
    w, h = read_png_size(path)
    target = str(body.get("target", "4k"))
    plan = upscale_plan(w, h, target)
    label = f"Upscale {target.upper()} · {UPSCALE_MODELS[model][1]}"
    return {"mode": "upscale", "prompt": label, "src": src, "model": model, "target": target, "width": plan["out_w"], "height": plan["out_h"],
            "plan": plan, "src_size": [w, h], "refs": [src], "seeds": [], "label": "", "eta": round(eta_upscale(w, h, model, target))}


def validate_then(body: dict):
    then = body.get("then")
    if not then:
        return None
    up = then.get("upscale") if isinstance(then, dict) else None
    if not up:
        return None
    if not isinstance(up, dict):
        raise ValueError("bad 'then.upscale' options")
    model, target = str(up.get("model", "quality")), str(up.get("target", "4k"))
    if model not in UPSCALE_MODELS or target not in UPSCALE_TARGETS:
        raise ValueError("bad 'then.upscale' options")
    if not upscale_available()[model]:
        raise ValueError(f"upscaler weights for '{model}' are missing")
    return {"model": model, "target": target}


RGBA_WRAP = "This is an RGBA image with transparency. {p}. The image has alpha channel and the background is transparent."


def validate(body: dict) -> dict:
    mode = body.get("mode", "generate")
    if mode == "upscale":
        return validate_upscale(body)
    if mode not in ("generate", "edit"):
        raise ValueError("mode must be generate, edit or upscale")
    prompt = body.get("prompt")
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not prompt:
        raise ValueError("prompt is empty")
    if len(prompt) > MAX_PROMPT or "\0" in prompt:
        raise ValueError(f"prompt must be plain text of at most {MAX_PROMPT} characters")
    model = REG.get(body.get("model"))
    if model is None:
        raise ValueError("unknown model")
    if not model_ready(model):
        raise ValueError(f"model '{model['name']}' is not ready (still downloading or files are missing)")
    fam = M.FAMILIES[model["family"]]
    caps = {**fam["caps"], **(model.get("caps") or {})}
    mult = fam["multiple"]
    w, h = as_int(body.get("width"), "width", 1, 100_000, 768), as_int(body.get("height"), "height", 1, 100_000, 768)
    for v in (w, h):
        if v % mult:
            raise ValueError(f"width and height must be divisible by {mult} for {fam['label']}")
        if not fam["min_side"] <= v <= fam["max_side"]:
            raise ValueError(f"each side must be {fam['min_side']}..{fam['max_side']} px for {fam['label']}")
    if w * h > CFG["max_pixels"]:
        raise ValueError("image too large: width*height must be <= 2048*2048")
    if mode == "edit" and w * h > CFG["max_pixels_edit"]:
        raise ValueError("edit size too large: width*height must be <= 1024*1024 (editing doubles the sequence length)")
    refs: list[str] = []
    if mode == "edit":
        if not model_can_edit(model):
            raise ValueError(f"'{model['name']}' cannot edit images (needs an edit-capable model with its vision file)")
        if not isinstance(body.get("refs", []), list):
            raise ValueError("refs must be a list of file names")
        refs = [str(r) for r in body.get("refs", [])][:3]
        if not refs:
            raise ValueError("edit needs 1-3 reference images")
        for r in refs:
            if not safe_path(r).is_file():
                raise ValueError(f"reference not found: {r}")
    count = max(1, min(as_int(body.get("count"), "count", -10**6, 10**6, 1), 4 if mode == "edit" else 8))
    seed = as_int(body.get("seed"), "seed", -1, 2**62, -1)  # empty or -1: a random one
    seed0 = seed if seed >= 0 else int.from_bytes(os.urandom(3), "big")
    transparent = bool(body.get("transparent")) and bool(caps.get("alpha"))
    levels = M.effort_levels(model)
    effort = max(0, min(as_int(body.get("effort"), "effort", -10**6, 10**6, 1), len(levels) - 1))
    variant = fam["variants"].get(model.get("variant")) or next(iter(fam["variants"].values()))
    steps_override = None if variant.get("fixed_steps") else levels[effort].get("steps")
    chat = str(body.get("chat") or "default")
    return {"mode": mode, "prompt": prompt, "width": w, "height": h, "refs": refs, "then": validate_then(body), "transparent": transparent,
            "sd_prompt": RGBA_WRAP.format(p=prompt.rstrip(". ")) if transparent else prompt, "model": model["id"], "model_name": model["name"],
            "effort": effort, "steps_override": steps_override, "steps_total": variant["steps"] if variant.get("fixed_steps") or not steps_override else steps_override,
            "chat": chat if re.fullmatch(r"[a-z0-9]{1,16}", chat) else "default", "seeds": [seed0 + i for i in range(count)], "label": str(body.get("label", ""))[:80]}


def build_cmd(job: dict) -> list[str]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = "edit" if job["mode"] == "edit" else "img"
    prefix = OUT / f"{stamp}_{tag}_{slugify(job['prompt'])}_{job['seeds'][0]}"
    n = len(job["seeds"])
    job["glob"] = glob.escape(str(prefix)) + ("_*.png" if n > 1 else ".png")
    out = str(prefix) + ("_%02d.png" if n > 1 else ".png")
    model = REG.get(job["model"])
    cmd = [*time_prefix(), sd_cli(), *M.sd_args(model, job["mode"], job.get("steps_override"))]
    if job["mode"] == "edit":
        for r in job["refs"]:
            cmd += ["-r", str(safe_path(r))]
    text = job.get("sd_prompt", job["prompt"])
    cmd += ["-p", " " + text if text.startswith("-") else text, "-W", str(job["width"]), "-H", str(job["height"]),
            "-s", str(job["seeds"][0]), "-b", str(n), *CFG["extra_args"], "-o", out]
    return cmd


def _stream(job: dict, cmd: list[str]) -> tuple[int, list[str]]:
    """Run one sd-cli process, parsing its progress into the job record."""
    proc = spawn(cmd)
    CURRENT.update(id=job["id"], proc=proc)
    tail: list[str] = []
    buf = b""
    upscaling = job["mode"] == "upscale"
    while True:
        chunk = proc.stdout.read(512)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            line = ANSI.sub("", raw.decode("utf-8", "replace")).strip()
            if not line:
                continue
            if "peak memory footprint" in line or line.startswith("[") or "error" in line.lower():
                tail.append(line)
                del tail[:-40]
            m = STEP_RE.search(line)
            if m:
                job.update(step=int(m.group(1)), steps=int(m.group(2)), phase="upscaling" if upscaling else "denoising")
                if not upscaling:
                    job["step_seconds"] = float(m.group(3))
                    if m.group(1) == m.group(2):
                        job["phase"] = "decoding"
                continue
            if upscaling:
                continue
            if "get_learned_condition completed" in line:
                job["phase"] = "denoising"
            elif "loading tensors completed" in line and job["phase"] == "loading model":
                job["phase"] = "encoding prompt"
            elif "decode_first_stage" in line or "decoded, taking" in line:
                job["phase"] = "saving"
            g = IMG_RE.search(line)
            if g:
                job.update(image_index=int(g.group(1)), image_total=int(g.group(2)), step=0)
            p = PEAK_RE.search(line)
            if p:
                job["peak_gb"] = round(int(p.group(1)) / 1e9, 1)
    rc = proc.wait()
    CURRENT.update(id=None, proc=None)
    return rc, tail


def _finish(job: dict, rc: int, tail: list[str], images: list[Path]) -> bool:
    job["log"] = tail[-12:]
    job["finished"] = time.time()
    job["seconds"] = round(job["finished"] - job["started"], 1)
    job["images"] = [p.name for p in images]
    if job.get("status") == "cancelled":
        return False
    if rc != 0 or not images:
        job["status"] = "error"
        job["error"] = next((ln for ln in reversed(tail) if "[E]" in ln or "rror" in ln), tail[-1] if tail else f"exit code {rc}")
        return False
    job["status"] = "done"
    return True


def enqueue(spec: dict, front: bool = False) -> dict:
    spec["eta"] = spec.get("eta") or round(eta_seconds(spec["width"], spec["height"], len(spec["seeds"]), spec["mode"] == "edit", REG.get(spec.get("model"))))
    job = {"id": uuid.uuid4().hex[:12], "status": "queued", "queued": time.time(), **spec}
    with LOCK:
        if not front and len(QUEUE) >= MAX_QUEUED:
            raise ApiError(f"the queue is full ({MAX_QUEUED} jobs): wait for some to finish or cancel a few", 429)
        JOBS[job["id"]] = job
        JOB_ORDER.append(job["id"])
        QUEUE.insert(0, job["id"]) if front else QUEUE.append(job["id"])
        while len(JOB_ORDER) > MAX_JOBS_KEPT:  # forget the oldest finished jobs
            old = next((i for i in JOB_ORDER if JOBS[i]["status"] in ("done", "error", "cancelled")), None)
            if old is None:
                break
            JOB_ORDER.remove(old)
            del JOBS[old]
    WAKE.set()
    return job


def run_upscale(job: dict) -> None:
    src = safe_path(job["src"])
    plan, model = job["plan"], job["model"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = re.sub(r"^\d{8}-\d{6}_(img_|edit_|composite_)?", "", Path(job["src"]).stem)
    final = OUT / f"{stamp}_up_{slugify(base)[:28]}_{plan['out_w']}x{plan['out_h']}.png"
    resize = plan["out_long"] != plan["native_long"]
    native = final.with_suffix(".native.png") if resize else final
    cmd = [*time_prefix(), sd_cli(), "-M", "upscale", "-i", str(src), "--upscale-model", str(upscale_model_path(model)),
           "--upscale-repeats", str(plan["repeats"]), "-o", str(native)]
    rc, tail = _stream(job, cmd)
    if rc == 0 and native.exists() and resize and job.get("status") != "cancelled":
        job["phase"] = "resizing"
        resize_long_side(native, final, plan["out_long"])
        native.unlink(missing_ok=True)
    if not _finish(job, rc, tail, [final] if final.exists() else []):
        native.unlink(missing_ok=True)
        return
    w, h = png_size(final.read_bytes()[:64])
    with LOCK:
        parent = next((r for r in HIST if r["file"] == job["src"]), None)
    add_rows([{
        "id": uuid.uuid4().hex[:10], "file": final.name, "kind": "upscale", "prompt": (parent or {}).get("prompt", ""), "width": w, "height": h,
        "seed": (parent or {}).get("seed"), "seconds": job["seconds"], "ts": int(job["finished"]), "parent": job["src"], "refs": [job["src"]],
        "fav": False, "job": job["id"], "up": {"model": model, "target": job["target"]}, "chat": (parent or {}).get("chat", "default"),
    }])


def run_job(job: dict) -> None:
    job.update(status="running", started=time.time(), step=0, steps=job.get("steps_total") or CFG["steps"], phase="loading model")
    if job["mode"] == "upscale":
        job["steps"] = 0
        return run_upscale(job)
    rc, tail = _stream(job, build_cmd(job))
    images = sorted(Path(p) for p in glob.glob(job["glob"]))
    if not _finish(job, rc, tail, images):
        return
    add_rows([{
        "id": uuid.uuid4().hex[:10], "file": p.name, "kind": "edit" if job["mode"] == "edit" else "gen",
        "prompt": job["prompt"], "width": job["width"], "height": job["height"], "seed": job["seeds"][0] + i,
        "seconds": job["seconds"], "ts": int(job["finished"]), "parent": job["refs"][0] if job["refs"] else None,
        "refs": job["refs"], "fav": False, "job": job["id"], "transparent": bool(job.get("transparent")),
        "model": job.get("model"), "chat": job.get("chat", "default"),
    } for i, p in enumerate(images)])
    if len(images) == 1 and job.get("model"):
        record_stat(job["model"], "edit" if job["mode"] == "edit" else "gen", job["width"] * job["height"], job["seconds"])
    then = job.get("then")
    if then:  # chained step, e.g. "create small, then upscale to 4K": runs next, even if the browser tab is closed
        nxt = []
        for p in images:
            try:
                spec = validate_upscale({"mode": "upscale", "src": p.name, "model": then["model"], "target": then["target"]})
                nxt.append(enqueue(spec, front=True)["id"])
            except ValueError as exc:
                job["error"] = f"auto-upscale skipped: {exc}"
        job["next"] = nxt


def worker() -> None:
    while True:
        WAKE.wait()
        while True:
            with LOCK:
                if not QUEUE:
                    WAKE.clear()
                    break
                job = JOBS[QUEUE.pop(0)]
            if job.get("status") == "cancelled":
                continue
            try:
                run_job(job)
            except OSError as exc:  # the engine vanished or cannot be started
                CURRENT.update(id=None, proc=None)
                job.update(status="error", error=f"could not start sd-cli: {exc.strerror or exc}")
            except Exception as exc:  # noqa: BLE001 - surface any failure to the client
                CURRENT.update(id=None, proc=None)
                traceback.print_exc()
                job.update(status="error", error=f"{type(exc).__name__}: {exc}")


def public(job: dict) -> dict:
    keys = ("id", "status", "mode", "prompt", "label", "width", "height", "seeds", "refs", "images", "step", "steps", "phase",
            "seconds", "peak_gb", "error", "log", "queued", "image_index", "image_total", "step_seconds", "eta", "src", "model", "target", "next", "transparent", "model_name", "effort", "chat")
    out = {k: job[k] for k in keys if k in job}
    if job.get("status") == "running":
        out["elapsed"] = round(time.time() - job["started"], 1)
    if job.get("status") == "queued":
        with LOCK:
            out["position"] = QUEUE.index(job["id"]) + 1 if job["id"] in QUEUE else 0
    return out


# ----------------------------------------------------------------------------------------- export
EXPORT_FORMATS = ("png", "webp", "jpeg", "svg")
SVG_PRESETS = {  # vtracer parameters; the first three trace colours, "bw" traces a one-colour silhouette
    "logo": {"colormode": "color", "filter_speckle": 8, "color_precision": 5, "layer_difference": 24, "corner_threshold": 60, "path_precision": 5},
    "illustration": {"colormode": "color", "filter_speckle": 4, "color_precision": 6, "layer_difference": 16, "corner_threshold": 60, "path_precision": 6},
    "detailed": {"colormode": "color", "filter_speckle": 2, "color_precision": 8, "layer_difference": 8, "corner_threshold": 60, "path_precision": 7},
    "bw": {"colormode": "binary", "filter_speckle": 4, "color_precision": 6, "layer_difference": 16, "corner_threshold": 60, "path_precision": 6},
}


def export_file(rel: str, fmt: str, quality: int = 90, preset: str = "illustration", max_side: int = 0, lossless: bool = False) -> dict:
    """Write a converted copy next to the library (in _exports/) and describe it. The original is never touched."""
    if Image is None:
        raise ValueError("export needs Pillow (bundled with the app)")
    if fmt not in EXPORT_FORMATS:
        raise ValueError("format must be one of " + ", ".join(EXPORT_FORMATS))
    src = safe_path(rel)
    if not src.is_file():
        raise ValueError("file not found")
    try:
        return _export(src, rel, fmt, max(1, min(int(quality), 100)), preset, max(0, min(int(max_side), MAX_IMAGE_SIDE)), lossless)
    except (Image.UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, OSError) as exc:
        raise ValueError(f"cannot convert this picture ({type(exc).__name__})") from exc


def _export(src: Path, rel: str, fmt: str, quality: int, preset: str, max_side: int, lossless: bool) -> dict:
    folder = OUT / "_exports"
    folder.mkdir(exist_ok=True)
    stem = re.sub(r"[^\w.-]+", "_", Path(rel).stem)[:60]
    ext = {"jpeg": "jpg"}.get(fmt, fmt)
    dst = folder / f"{stem}.{ext}"
    n = 1
    while dst.exists():
        n += 1
        dst = folder / f"{stem}-{n}.{ext}"
    part = folder / f".{dst.name}.part"  # written under this name and renamed when complete
    try:
        info = _convert(src, part, fmt, quality, preset, max_side, lossless, folder, stem)
        os.replace(part, dst)
    finally:
        part.unlink(missing_ok=True)
    w, h = (info.get("traced") or [0, 0]) if fmt == "svg" else (0, 0)
    if fmt != "svg":
        with Image.open(dst) as out:
            w, h = out.size
    return {"file": str(dst.relative_to(OUT)).replace(os.sep, "/"), "format": fmt, "bytes": dst.stat().st_size, "width": w, "height": h, **info}


def _convert(src: Path, dst: Path, fmt: str, quality: int, preset: str, max_side: int, lossless: bool, folder: Path, stem: str) -> dict:
    with open_image(src) as im:
        im.load()
        has_alpha = im.mode in ("RGBA", "LA") or "transparency" in im.info
        if max_side and max(im.size) > max_side and fmt != "svg":
            k = max_side / max(im.size)
            im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
        info: dict = {"alpha": has_alpha and fmt in ("png", "webp", "svg")}
        if fmt == "png":
            im.convert("RGBA" if has_alpha else "RGB").save(dst, "PNG", optimize=True)
        elif fmt == "webp":
            im.convert("RGBA" if has_alpha else "RGB").save(dst, "WEBP", quality=int(quality), lossless=lossless, method=4)
        elif fmt == "jpeg":
            base = im.convert("RGBA")
            flat = Image.new("RGB", base.size, (255, 255, 255))
            flat.paste(base, mask=base.getchannel("A"))
            flat.save(dst, "JPEG", quality=int(quality), optimize=True, subsampling=0 if quality >= 92 else 2)
        else:
            try:
                import vtracer  # vector tracing: great for logos and flat art, huge and messy for photographs
            except ImportError:
                raise ValueError("SVG export needs the vtracer package (pip install vtracer)") from None
            params = SVG_PRESETS.get(preset)
            if params is None:
                raise ValueError("preset must be one of " + ", ".join(SVG_PRESETS))
            limit = max_side or 1536  # tracing cost and file size grow with the pixel count
            work = im.convert("RGBA" if has_alpha else "RGB")
            if max(work.size) > limit:
                k = limit / max(work.size)
                work = work.resize((round(work.width * k), round(work.height * k)), Image.LANCZOS)
            tmp = folder / f".{stem}.trace.png"
            work.save(tmp, "PNG")
            try:
                vtracer.convert_image_to_svg_py(str(tmp), str(dst), hierarchical="stacked", mode="spline", length_threshold=4.0,
                                                max_iterations=10, splice_threshold=45, **params)
            finally:
                tmp.unlink(missing_ok=True)
            text = dst.read_text(encoding="utf-8-sig")
            info.update(paths=text.count("<path"), traced=list(work.size))
    return info


# ----------------------------------------------------------------------------------------- first-run setup
HF = "https://huggingface.co"
GH = "https://github.com"
SETUP: dict = {"running": False, "cancel": False, "items": {}}


class Cancelled(Exception):
    pass


def components() -> list[dict]:
    """Everything the app needs besides its own code. Sizes are the exact upstream byte counts."""
    q8 = "Qwen-Image-2.1-Turbo-Q8_0.gguf"
    enc = "Qwen-Image-2.1-Turbo-Abliterated-Uncensored-Q8_0.gguf"
    mm = "mmproj-Qwen-Image-2.1-Turbo-Abliterated-Uncensored-F16.gguf"
    up = Path(CFG["upscale_dir"])
    return [
        {"id": "engine", "group": "engine", "label": "stable-diffusion.cpp engine", "note": "bundled with the app; downloaded only if missing", "kind": "engine", "size": 33_000_000, "required": True},
        {"id": "diffusion", "group": "models", "label": "Qwen-Image 2.1 Turbo · denoiser Q8", "note": "main model", "url": f"{HF}/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main/{q8}", "path": Path(CFG["diffusion"]), "size": 7_591_554_784, "required": True},
        {"id": "llm", "group": "models", "label": "Text encoder Q8 (no refusals)", "note": "understands prompts, including Russian", "url": f"{HF}/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF/resolve/main/{enc}", "path": Path(CFG["llm"]), "size": 8_709_520_096, "required": True},
        {"id": "vae", "group": "models", "label": "VAE (image decoder)", "note": "", "url": f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors", "path": Path(CFG["vae"]), "size": 675_509_688, "required": True},
        {"id": "mmproj", "group": "models", "label": "Vision module (photo editing)", "note": "needed for editing and the brush", "url": f"{HF}/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF/resolve/main/{mm}", "path": Path(CFG["mmproj"]), "size": 1_159_029_984, "required": False},
        {"id": "up-quality", "group": "upscale", "label": "Upscaler: Quality (Real-ESRGAN x4plus)", "note": "BSD-3", "url": f"{GH}/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", "path": up / "RealESRGAN_x4plus.pth", "size": 67_040_989, "required": False},
        {"id": "up-fast", "group": "upscale", "label": "Upscaler: Fast (4xNomos2 MoSR)", "note": "CC-BY, for drafts", "url": f"{GH}/Phhofm/models/releases/download/4xNomos2_hq_mosr/4xNomos2_hq_mosr.pth", "path": up / "4xNomos2_hq_mosr.pth", "size": 17_213_494, "required": False},
        {"id": "up-illustration", "group": "upscale", "label": "Upscaler: Illustration (anime 6B)", "note": "BSD-3", "url": f"{GH}/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth", "path": up / "RealESRGAN_x4plus_anime_6B.pth", "size": 17_938_799, "required": False},
    ]


def comp_present(c: dict) -> bool:
    if c["id"] == "engine":
        return Path(sd_cli()).is_file()
    p = c["path"]
    return p.is_file() and p.stat().st_size >= c["size"] * 0.99


def comp_partial(c: dict) -> int:
    if c["id"] == "engine":
        return 0
    part = c["path"].with_name(c["path"].name + ".part")
    return part.stat().st_size if part.exists() else 0


HF_TOKEN_PATH = DATA / "hf_token"
TOKEN_RE = re.compile(r"[A-Za-z0-9_.~+/=-]{8,512}")  # no whitespace or control characters: the value goes into an HTTP header
ENGINE_RELEASE = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
ENGINE_URL_PREFIX = "https://github.com/leejet/stable-diffusion.cpp/releases/download/"
ENGINE_SIZE = 33_000_000  # the download (bytes), as shown in the setup list
MAX_ENGINE_ZIP, MAX_ENGINE_UNPACKED, MAX_ENGINE_ENTRIES = 300_000_000, 1_000_000_000, 200  # the real archives: 35 MB, 90 MB, 5-21 files
ENGINE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,99}")


def hf_token() -> str | None:
    try:
        tok = HF_TOKEN_PATH.read_text(encoding="utf-8-sig").strip()
    except (OSError, ValueError):
        return None
    return tok if TOKEN_RE.fullmatch(tok) else None


def set_hf_token(tok: str) -> None:
    """Store (or, when empty, delete) the Hugging Face token, readable by this user only from the first byte written."""
    if not tok:
        HF_TOKEN_PATH.unlink(missing_ok=True)
        return
    if not TOKEN_RE.fullmatch(tok):
        raise ValueError("that does not look like a Hugging Face token (letters, digits and _-. only, no spaces)")
    M.atomic_write(HF_TOKEN_PATH, tok, mode=0o600)


def tighten_token_file() -> None:
    try:
        if SYSTEM != "Windows" and HF_TOKEN_PATH.stat().st_mode & 0o077:
            HF_TOKEN_PATH.chmod(0o600)
    except OSError:
        pass


def _http(url: str, headers: dict | None = None):
    """GET with our rules: https to known hosts only, the token goes to huggingface.co only, and redirects (to CDNs) never
    carry it along (see models.SafeRedirect)."""
    return M.open_url(url, {"User-Agent": "ImageLoom/" + VERSION, **(headers or {})}, token=hf_token(), timeout=60)


def download_file(url: str, dest: Path, expected: int, st: dict) -> None:
    """Resumable download (HTTP Range) with progress; the .part file survives interruptions and restarts and is renamed to
    `dest` only after the size checks. A server that sends far more than `expected` is cut off."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and expected and dest.stat().st_size >= expected * 0.99:  # already complete (retry after another file failed)
        st.update(total=expected, bytes=dest.stat().st_size)
        return
    part = dest.with_name(dest.name + ".part")
    limit = int(expected * 1.05) + 16_000_000 if expected > 1_000_000 else MAX_FILE_BYTES
    fails = 0
    while True:
        pos = part.stat().st_size if part.exists() else 0
        try:
            with _http(url, {"Range": f"bytes={pos}-"} if pos else None) as r:
                if pos and r.status == 200:  # the server ignored Range: start over
                    pos = 0
                    part.unlink(missing_ok=True)
                cr = r.headers.get("Content-Range")
                total = int(cr.rsplit("/", 1)[1]) if cr and "/" in cr else pos + int(r.headers.get("Content-Length") or 0)
                if total > limit:
                    raise RuntimeError(f"the server announced {total} bytes, far more than the {expected} expected")
                st.update(total=total or expected, bytes=pos)
                last_t, last_b = time.time(), pos
                with open(part, "ab" if pos else "wb") as fh:
                    while True:
                        if SETUP["cancel"]:
                            raise Cancelled()
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        st["bytes"] += len(chunk)
                        if st["bytes"] > limit:
                            fh.close()
                            part.unlink(missing_ok=True)
                            raise RuntimeError("the server sent more data than expected")
                        now = time.time()
                        if now - last_t >= 1:
                            st["speed"] = (st["bytes"] - last_b) / (now - last_t)
                            last_t, last_b = now, st["bytes"]
            break
        except (Cancelled, RuntimeError):
            raise
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and pos:  # nothing left to fetch: the part file is already complete
                break
            fails += 1
            if fails > 6 or exc.code in (401, 403, 404) or 300 <= exc.code < 400:  # (a redirect we refused is final)
                raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
            time.sleep(min(4 * fails, 30))
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, http.client.HTTPException) as exc:  # (incl. a connection dropped mid-transfer: resume)
            if getattr(exc, "errno", None) == errno.ENOSPC:
                raise RuntimeError("the disk is full") from exc
            fails += 1
            if fails > 8:
                raise RuntimeError(f"network error: {exc}") from exc
            st["speed"] = 0
            time.sleep(min(4 * fails, 30))
    size = part.stat().st_size
    if st.get("total") and size < st["total"]:
        raise RuntimeError("download is incomplete")
    if expected and size < expected * 0.99:
        raise RuntimeError("downloaded file is smaller than expected")
    part.replace(dest)


def engine_supported() -> bool:
    machine = platform.machine().lower()
    return (SYSTEM == "Darwin" and "arm" in machine) or SYSTEM == "Windows" or (SYSTEM == "Linux" and machine in ("x86_64", "amd64"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install_engine_zip(zpath: Path, eng: Path) -> None:
    """Unpack the engine archive flat into `eng`: only plain file names are used (no directories at all, so no zip-slip),
    entry count and unpacked size are bounded, and files are moved into place only after everything extracted."""
    stage = eng / ".unpack"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    names: list[str] = []
    try:
        with zipfile.ZipFile(zpath) as z:
            for i in z.infolist():  # a legitimate archive never names a parent folder, an absolute path or a drive: tampered, refuse it
                n = i.filename.replace("\\", "/")
                if ".." in n.split("/") or n.startswith("/") or re.match(r"[A-Za-z]:", n) or "\0" in n:
                    raise RuntimeError(f"the engine archive contains an unsafe path ({n[:60]!r})")
            infos = [i for i in z.infolist() if not i.is_dir()]
            if len(infos) > MAX_ENGINE_ENTRIES or sum(i.file_size for i in infos) > MAX_ENGINE_UNPACKED:
                raise RuntimeError("the engine archive is unexpectedly large")
            left = MAX_ENGINE_UNPACKED  # counts real bytes too, in case an entry lies about its size
            for info in infos:
                name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
                if not ENGINE_FILE.fullmatch(name) or name.endswith((".txt", ".md")) or M.RESERVED_NAME.fullmatch(name):
                    continue
                with z.open(info) as src, open(stage / name, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        left -= len(chunk)
                        if left < 0:
                            raise RuntimeError("the engine archive is unexpectedly large")
                        dst.write(chunk)
                names.append(name)
        if not any(n.startswith("sd-cli") for n in names):
            raise RuntimeError("the engine archive does not contain sd-cli")
        for name in names:
            if SYSTEM != "Windows" and name in ("sd-cli", "sd-server"):
                (stage / name).chmod(0o755)
            os.replace(stage / name, eng / name)
    except zipfile.BadZipFile:
        raise RuntimeError("the downloaded engine archive is damaged") from None
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def download_engine(st: dict) -> None:
    """Fetch the sd-cli build for this OS from the official stable-diffusion.cpp release, verify it and unpack it."""
    if not engine_supported():
        raise RuntimeError(f"no prebuilt engine for {SYSTEM}/{platform.machine()}: install sd-cli yourself and point IMAGELOOM_SD_CLI at it")
    with _http(ENGINE_RELEASE) as r:
        rel = json.loads(r.read(5_000_000))
    pat = {"Darwin": r"Darwin-macOS.*(arm64)\.zip$", "Windows": r"bin-win-vulkan-x64\.zip$", "Linux": r"Linux-Ubuntu.*x86_64-vulkan\.zip$"}[SYSTEM]
    asset = next((a for a in rel.get("assets", []) if isinstance(a, dict) and re.search(pat, str(a.get("name")))), None)
    if asset is None:
        raise RuntimeError(f"the latest stable-diffusion.cpp release has no build for {SYSTEM}")
    url, size = asset.get("browser_download_url"), asset.get("size")
    if not (isinstance(url, str) and url.lower().startswith(ENGINE_URL_PREFIX) and isinstance(size, int) and 0 < size <= MAX_ENGINE_ZIP):
        raise RuntimeError("the engine release metadata looks wrong")
    eng = DATA / "engine"
    eng.mkdir(parents=True, exist_ok=True)
    zpath = eng / "_engine.zip"
    st["label"] = str(asset["name"])[:100]
    download_file(url, zpath, size, st)
    digest = str(asset.get("digest") or "").removeprefix("sha256:").lower()
    if digest and sha256_of(zpath) != digest:
        zpath.unlink(missing_ok=True)
        raise RuntimeError("engine archive failed its sha256 check")
    try:
        install_engine_zip(zpath, eng)
    finally:
        zpath.unlink(missing_ok=True)


def run_downloads(items: list[dict], after=None) -> None:
    """Sequentially download `items` ({id, kind, url, path, size}); `after(ok)` runs when the batch ends."""
    ok = True
    try:
        for it in items:
            if SETUP["cancel"]:
                ok = False
                break
            st = SETUP["items"][it["id"]] = {"state": "downloading", "bytes": 0, "total": it["size"], "speed": 0, "error": None}
            try:
                if it.get("kind") == "engine":
                    download_engine(st)
                else:
                    download_file(it["url"], Path(it["path"]), it["size"], st)
                st["state"] = "done"
            except Cancelled:
                st["state"] = "cancelled"
                ok = False
            except Exception as exc:  # noqa: BLE001
                st.update(state="error", error=str(exc))
                ok = False
    finally:
        SETUP["running"] = False
        if after:
            try:
                after(ok)
            except Exception:  # noqa: BLE001
                traceback.print_exc()


def start_downloads(items: list[dict], after=None, register=None) -> None:
    """Claim the single download slot (atomically: two requests cannot both start), run `register()` while holding it, then
    download `items` on a background thread."""
    with LOCK:
        if SETUP["running"]:
            raise ValueError("a download is already running: wait until it finishes")
        SETUP.update(running=True, cancel=False)
    try:
        if register:
            register()
        SETUP["items"] = {i["id"]: {"state": "queued", "bytes": 0, "total": i["size"], "speed": 0, "error": None} for i in items}
    except BaseException:
        SETUP["running"] = False
        raise
    threading.Thread(target=run_downloads, args=(items, after), daemon=True).start()


def setup_status() -> dict:
    items = []
    for c in components():
        st = SETUP["items"].get(c["id"], {})
        items.append({"id": c["id"], "group": c["group"], "label": c["label"], "note": c["note"], "size": c["size"], "required": c["required"],
                      "present": comp_present(c), "partial": comp_partial(c), "state": st.get("state", "idle"), "bytes": st.get("bytes", 0),
                      "total": st.get("total", c["size"]), "speed": st.get("speed", 0), "error": st.get("error")})
    free = shutil.disk_usage(DATA).free
    need = sum(max(i["size"] - i["partial"], 0) for i in items if not i["present"])
    return {"components": items, "running": SETUP["running"], "free_bytes": free, "missing_bytes": need, "data_dir": str(DATA),
            "complete": models_ready(), "edit_ready": edit_ready(), "system": SYSTEM}


def setup_start(ids: list[str]) -> dict:
    if SETUP["running"]:
        raise ValueError("a download is already running")
    comps = {c["id"]: c for c in components()}
    ids = [i for i in ids if i in comps and not comp_present(comps[i])]
    if not ids:
        raise ValueError("nothing to download")
    need = sum(max(comps[i]["size"] - comp_partial(comps[i]), 0) for i in ids)
    free = shutil.disk_usage(DATA).free
    if need + 2_000_000_000 > free:
        raise ValueError(f"not enough disk space: need {need / 1e9:.1f} GB plus a 2 GB reserve, free {free / 1e9:.1f} GB")
    items = [{"id": i, "kind": comps[i].get("kind"), "url": comps[i].get("url"), "path": comps[i].get("path"), "size": comps[i]["size"]} for i in ids]
    start_downloads(items)
    return setup_status()


# ----------------------------------------------------------------------------------------- models (Hugging Face links)
def public_models() -> dict:
    out = []
    for m in REG.all():
        pm = M.public_model(m, model_ready(m))
        pm["can_edit"] = model_can_edit(m)
        if not m.get("builtin"):
            pm["downloading"] = {k: v for k, v in SETUP["items"].items() if k.startswith(f"m:{m['id']}:")}
        out.append(pm)
    return {"models": out, "active": REG.data.get("active"), "running": SETUP["running"], "token": bool(hf_token()), "free_bytes": shutil.disk_usage(DATA).free}


def inside(root: Path, path: Path | str) -> bool:
    return root.resolve() in Path(path).resolve().parents


def models_add(body: dict) -> dict:
    """Create a registry entry from a plan returned by /api/models/inspect and start downloading its files (and the sd-cli
    engine first, when it is not installed yet: the model only turns ready once everything arrived)."""
    if SETUP["running"]:
        raise ValueError("another download is running: wait until it finishes")
    if len(REG.data["models"]) >= MAX_MODELS:
        raise ValueError(f"at most {MAX_MODELS} models can be registered: remove some first")
    family, variant = str(body.get("family") or "custom"), str(body.get("variant") or "base")
    if family not in M.FAMILIES:
        raise ValueError("unknown family")
    fam = M.FAMILIES[family]
    if variant not in fam["variants"]:
        variant = next(iter(fam["variants"]))
    plan = M.inspect_repo(str(body.get("repo", "")), hf_token())
    repo = plan["repo"]
    cand = next((c for c in plan["candidates"] if c["path"] == body.get("file")), None)
    if cand is None:
        raise ValueError("choose one of the weight files offered for this repository")
    fname = M.safe_filename(cand["path"].rsplit("/", 1)[-1])
    if not fname.lower().endswith((".gguf", ".safetensors", ".sft")) or not 1_000_000 <= cand["size"] <= MAX_FILE_BYTES:
        raise ValueError("this file does not look like model weights (type or size)")
    name = re.sub(r"[\x00-\x1f]", "", str(body.get("name") or plan["name"]))[:80].strip() or plan["name"]
    mid = M.slug(f"{name}-{cand['quant'] or 'w'}")
    if (REG.get(mid) or {}).get("builtin") or mid in (MODELS.name, "upscale"):
        raise ValueError("a built-in model has this id: choose another name")
    root = DATA / "models"
    folder, shared = root / mid, root / "_shared"
    role_key = "checkpoint" if fam.get("single") else "diffusion"
    files = {role_key: str(folder / fname)}
    items = [{"id": f"m:{mid}:{role_key}", "url": M.file_url(repo, cand["path"]), "path": files[role_key], "size": cand["size"]}]
    for c in plan["companions"]:
        if c.get("optional") and not body.get("with_optional", True):
            continue
        dest = shared / M.safe_filename(c["file"])
        files[c["role"]] = str(dest)
        if not (dest.is_file() and c["size"] and dest.stat().st_size >= c["size"] * 0.99):
            items.append({"id": f"m:{mid}:{c['role']}", "url": c["url"], "path": str(dest), "size": min(c["size"] or 1, MAX_FILE_BYTES)})
    if not all(inside(root, p) for p in files.values()):
        raise ValueError("refusing to write outside the models folder")
    model_bytes = sum(i["size"] for i in items)
    if not Path(sd_cli()).is_file():  # a clean install: the first job would fail without the engine
        if not engine_supported():
            raise ValueError("no prebuilt sd-cli engine for this system: install sd-cli yourself and set IMAGELOOM_SD_CLI")
        items.insert(0, {"id": f"m:{mid}:engine", "kind": "engine", "size": ENGINE_SIZE})
    need = sum(max(i["size"] - (Path(i["path"] + ".part").stat().st_size if i.get("path") and Path(i["path"] + ".part").exists() else 0), 0) for i in items)
    free = shutil.disk_usage(DATA).free
    if need + 2_000_000_000 > free:
        raise ValueError(f"not enough disk space: need {need / 1e9:.1f} GB plus a 2 GB reserve, free {free / 1e9:.1f} GB")
    entry = {"id": mid, "name": name, "family": family, "variant": variant, "files": files, "status": "downloading", "source": f"{M.HF}/{repo}",
             "license": str(plan.get("license", ""))[:80], "tags": [fam["label"], cand["quant"]] if cand["quant"] else [fam["label"]], "added": int(time.time()),
             "size": model_bytes, "experimental": family != "qwen-image-2.1", "verified": False, "quant": cand["quant"]}

    def done(ok: bool) -> None:
        REG.update(mid, status="ready" if ok and Path(sd_cli()).is_file() else "error")

    start_downloads(items, done, register=lambda: REG.add(entry))
    return public_models()


def models_delete(mid: str) -> dict:
    """Remove a registered model and move its files to the Trash. Only files inside <data>/models that no other model (the
    built-in one included) uses are touched: a registry entry can never make this delete something elsewhere."""
    m = REG.get(mid)
    if m is None or m.get("builtin"):
        raise ValueError("this model cannot be removed")
    REG.remove(mid)
    keep = {Path(p).resolve() for other in REG.all() for p in (other.get("files") or {}).values()}
    root = DATA / "models"
    for p in (m.get("files") or {}).values():
        if not inside(root, p):
            continue
        path = Path(p)
        if path.resolve() not in keep and path.is_file():
            move_to_trash(path)
        path.with_name(path.name + ".part").unlink(missing_ok=True)
    folder = root / M.slug(mid)
    if inside(root, folder) and folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
    return public_models()


# ----------------------------------------------------------------------------------------- chats
def load_chats() -> dict:
    data = M.read_json(CHATS_PATH, {"chats": []})
    chats = [c for c in data.get("chats") or [] if isinstance(c, dict) and re.fullmatch(r"[a-z0-9]{1,16}", str(c.get("id")))] if isinstance(data.get("chats"), list) else []
    if not any(c["id"] == "default" for c in chats):
        chats.insert(0, {"id": "default", "title": "All work", "model": None, "effort": 1, "created": int(time.time()), "updated": int(time.time()), "archived": False})
    return {"chats": chats}


CHATS = load_chats()


def save_chats() -> None:
    M.atomic_write(CHATS_PATH, json.dumps(CHATS, ensure_ascii=False, indent=1))


def clean_text(v, limit: int) -> str:
    """Display text from a client: control characters removed, length bounded."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(v)).strip()[:limit]


def chat_field(k: str, v):
    if k == "title":
        return clean_text(v, 60)
    if k == "model":
        return v if v is None or (isinstance(v, str) and len(v) <= 80) else REG.data.get("active")
    return max(0, min(as_int(v, "effort", -10**6, 10**6, 1), 9))


def chats_api(body: dict, cid: str | None = None, delete: bool = False) -> dict:
    with LOCK:
        if cid is None:  # create
            if len(CHATS["chats"]) >= MAX_CHATS:  # archived chats are hidden anyway: drop them before refusing
                CHATS["chats"] = [c for c in CHATS["chats"] if not c.get("archived")]
                if len(CHATS["chats"]) >= MAX_CHATS:
                    raise ValueError(f"too many chats (the limit is {MAX_CHATS}): delete some first")
            cid = uuid.uuid4().hex[:8]
            CHATS["chats"].append({"id": cid, "title": chat_field("title", body.get("title") or "New chat"), "model": chat_field("model", body.get("model") or REG.data.get("active")),
                                   "effort": chat_field("effort", body.get("effort")), "created": int(time.time()), "updated": int(time.time()), "archived": False})
        else:
            c = next((c for c in CHATS["chats"] if c["id"] == cid), None)
            if c is None:
                raise ValueError("unknown chat")
            if delete:
                if cid == "default":
                    raise ValueError("the default chat cannot be removed")
                c["archived"] = True
            else:
                for k in ("title", "model", "effort"):
                    if k in body:
                        c[k] = chat_field(k, body[k])
                c["updated"] = int(time.time())
        save_chats()
        return {"chats": [c for c in CHATS["chats"] if not c.get("archived")], "id": cid}


# ----------------------------------------------------------------------------------------- HTTP
# The UI is one self-contained page: no external scripts, styles, fonts or connections are ever needed.
CSP = ("default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' blob: data:; "
       "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # talking to 127.0.0.1 must never go through a proxy


class Handler(BaseHTTPRequestHandler):
    server_version = f"imageloom/{VERSION}"
    sys_version = ""  # do not advertise the Python version
    timeout = 30  # a client that stalls mid-request (slowloris) must not hold a thread forever

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _send(self, code: int, payload, ctype="application/json", headers: dict | None = None) -> None:
        data = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith(("application/json", "text/html")) else "max-age=3600")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")  # (and no CORS headers anywhere: other sites can never read a reply)
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _guard(self, post: bool) -> bool:
        """Block DNS rebinding and cross-site requests: only our own UI/CLI may talk to this server."""
        host = (self.headers.get("Host") or "").lower()
        port = str(CFG["port"])
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self._send(403, {"error": "bad host"})
            return False
        origin = self.headers.get("Origin")
        if (origin and origin.lower() not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")) or self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._send(403, {"error": "cross-origin request blocked"})
            return False
        if post and self.headers.get("X-ImageLoom") != "1":
            self._send(403, {"error": "missing X-ImageLoom header"})
            return False
        return True

    def _body(self, limit=MAX_UPLOAD) -> bytes:
        if self.headers.get("Transfer-Encoding"):
            raise ApiError("chunked bodies are not supported: send a Content-Length", 411)
        raw = (self.headers.get("Content-Length") or "0").strip()
        if not (raw.isascii() and raw.isdigit() and len(raw) <= 12):  # also rejects negative and fractional lengths
            raise ValueError("bad Content-Length")
        length = int(raw)
        if length > limit:
            raise ApiError("request body too large", 413)
        data = self.rfile.read(length) if length else b""
        if len(data) != length:
            raise ValueError("incomplete request body")
        return data

    def _json(self) -> dict:
        raw = self._body(MAX_JSON)
        try:
            body = json.loads(raw or b"{}")
        except ValueError as exc:
            raise ValueError("invalid JSON") from exc
        if not isinstance(body, dict):
            raise ValueError("the request body must be a JSON object")
        return body

    def _file(self, path: Path, ctype: str) -> None:
        if not path.is_file():
            return self._send(404, {"error": "not found"})
        if path.stat().st_size > MAX_UPLOAD:
            return self._send(413, {"error": "file too large to serve"})
        self._send(200, path.read_bytes(), ctype, {"Content-Security-Policy": CSP, "X-Frame-Options": "DENY"} if ctype.startswith("text/html") else None)

    def _image(self, path: Path) -> None:
        ctype = IMG_TYPES[path.suffix.lower()]
        if not path.is_file():
            return self._send(404, {"error": "not found"})
        if path.stat().st_size > MAX_UPLOAD:
            return self._send(413, {"error": "file too large to serve"})
        # an SVG opened as a page must not run scripts
        self._send(200, path.read_bytes(), ctype, {"Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'"} if ctype == "image/svg+xml" else None)

    def _run(self, fn, *args) -> None:
        """Run a route; every failure becomes a JSON error (400 for bad input, never a traceback for the client)."""
        try:
            fn(*args)
        except ApiError as exc:
            self._send(exc.status, {"error": str(exc)})
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionError, TimeoutError):
            pass  # the client went away
        except Exception:  # noqa: BLE001
            traceback.print_exc()  # the details go to the server log
            self._send(500, {"error": "internal error (see the server log)"})

    def do_GET(self) -> None:  # noqa: N802
        if self._guard(False):
            self._run(self._route_get)

    def _route_get(self) -> None:
        url = urlparse(self.path)
        path = unquote(url.path)
        qs = parse_qs(url.query)
        if path in ("/", "/index.html"):
            return self._file(WEB / "index.html", "text/html; charset=utf-8")
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if path == "/api/status":
            with LOCK:
                queued = len(QUEUE)
            return self._send(200, {
                "ok": True, "version": VERSION, "model_ready": models_ready(), "edit_ready": edit_ready(),
                "engine": "stable-diffusion.cpp · Qwen-Image 2.1 Turbo Q8 · encoder without refusals",
                "queued": queued, "running": CURRENT["id"], "output_dir": str(OUT), "multiple": CFG["multiple"],
                "system": SYSTEM, "data_dir": str(DATA), "host": os.environ.get("IMAGELOOM_HOST", ""), "setup_needed": not any_model_ready(), "active_model": REG.data.get("active"), "engine_found": Path(sd_cli()).is_file(),
                "upscale": {k: {"ready": v, "label": UPSCALE_MODELS[k][1], "note": UPSCALE_MODELS[k][2]} for k, v in upscale_available().items()},
                "upscale_targets": list(UPSCALE_TARGETS), "max_upscale_side": CFG["max_upscale_side"],
                "max_pixels": CFG["max_pixels"], "warn_pixels": CFG["warn_pixels"], "max_pixels_edit": CFG["max_pixels_edit"], "warn_pixels_edit": CFG["warn_pixels_edit"], "min_side": CFG["min_side"], "max_side": CFG["max_side"],
            })

        def qint(name: str, lo: int, hi: int, default: int | None = None) -> int:
            return as_int((qs.get(name) or [None])[0], name, lo, hi, default)

        if path == "/api/eta" and qs.get("mode", [""])[0] == "upscale":
            model = qs.get("model", ["quality"])[0]
            if model not in UPSCALE_MODELS:
                raise ValueError("model must be one of " + ", ".join(UPSCALE_MODELS))
            w, h, target = qint("w", 1, 100_000), qint("h", 1, 100_000), qs.get("target", ["4k"])[0]
            plan = upscale_plan(w, h, target)
            return self._send(200, {"seconds": round(eta_upscale(w, h, model, target)), "out_w": plan["out_w"], "out_h": plan["out_h"], "repeats": plan["repeats"]})
        if path.startswith("/thumb/"):
            try:
                src = safe_path(path[len("/thumb/"):])
            except ValueError:
                return self._send(403, {"error": "forbidden"})
            if not src.is_file():
                return self._send(404, {"error": "not found"})
            size = -(-clamp_int((qs.get("s") or ["640"])[0], 128, 1600) // 64) * 64  # a few sizes only: the cache cannot be flooded with variants
            st = src.stat()
            if st.st_size < 1_200_000 or src.suffix.lower() == ".svg":
                return self._image(src)
            key = hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}|{size}".encode()).hexdigest()[:16]
            ext, ctype = ("webp", "image/webp") if Image is not None else ("jpg", "image/jpeg")
            out = THUMBS / f"{key}.{ext}"
            if not out.exists():
                with THUMB_SLOTS:
                    tmp = THUMBS / f"{key}.{threading.get_ident()}.tmp.{ext}"  # built under another name, then renamed: no half-written thumbnails
                    try:
                        if not out.exists() and make_thumb(src, tmp, size):
                            os.replace(tmp, out)
                    except Exception:  # noqa: BLE001
                        pass
                    finally:
                        tmp.unlink(missing_ok=True)
            if out.exists():
                return self._send(200, out.read_bytes(), ctype)
            return self._image(src)
        if path == "/api/eta":
            w, h, n = qint("w", 1, 100_000), qint("h", 1, 100_000), qint("n", 1, 1000, 1)
            sec, guess = eta_info(w, h, n, qs.get("edit", ["0"])[0] == "1", REG.get((qs.get("model") or [None])[0]))
            return self._send(200, {"seconds": round(sec), "guess": guess})
        if path == "/api/setup":
            return self._send(200, setup_status())
        if path == "/api/models":
            return self._send(200, public_models())
        if path == "/api/chats":
            with LOCK:
                return self._send(200, {"chats": [c for c in CHATS["chats"] if not c.get("archived")]})
        if path == "/api/history":
            return self._send(200, visible_history())
        if path == "/api/jobs":
            with LOCK:
                ids = JOB_ORDER[-25:]
            return self._send(200, [public(JOBS[i]) for i in ids if i in JOBS])
        m = re.fullmatch(r"/api/jobs/([a-f0-9]+)", path)
        if m:
            job = JOBS.get(m.group(1))
            return self._send(200, public(job)) if job else self._send(404, {"error": "unknown job"})
        if path.startswith("/img/"):
            try:
                target = safe_path(path[len("/img/"):])
            except ValueError:
                return self._send(403, {"error": "forbidden"})
            return self._image(target)
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._guard(True):
            return
        url = urlparse(self.path)
        self._run(self._route_post, url.path, parse_qs(url.query))

    def _hf(self, fn):
        """Call something that talks to Hugging Face and map its failures to HTTP answers."""
        try:
            return self._send(200, fn())
        except PermissionError as exc:
            return self._send(403, {"error": str(exc), "need_token": True})
        except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError, http.client.HTTPException) as exc:
            return self._send(502, {"error": f"could not reach Hugging Face: {getattr(exc, 'reason', exc)}"})

    def _route_post(self, path: str, qs: dict) -> None:
        if path == "/api/generate":
            body = self._json()
            if not Path(sd_cli()).is_file():
                return self._send(503, {"error": "the sd-cli engine is not installed yet: open the setup screen (or add a model, which fetches it) to download it", "need_engine": True})
            if body.get("mode") != "upscale" and not any_model_ready():
                return self._send(503, {"error": "no model is ready: open the setup screen (or run `imageloom doctor`)"})
            body.setdefault("model", REG.data.get("active"))
            return self._send(202, public(enqueue(validate(body))))
        if path in ("/api/upload", "/api/save"):
            data = self._body()
            w, h = check_png(data)
            if path == "/api/upload":
                name = UPLOADS / f"{hashlib.sha256(data).hexdigest()[:20]}.png"
                kind, prompt, parent = "upload", clean_text((qs.get("name") or ["upload"])[0], 80), None
            else:
                parent = (qs.get("parent") or [None])[0]
                try:
                    parent = parent if parent and safe_path(parent).is_file() else None
                except ValueError:
                    parent = None
                prompt = clean_text((qs.get("prompt") or ["composite"])[0], 500)
                kind = "composite"
                base = f"{time.strftime('%Y%m%d-%H%M%S')}_composite_{slugify(prompt)}"
                name, n = OUT / f"{base}.png", 1
                while name.exists():
                    n += 1
                    name = OUT / f"{base}-{n}.png"
            if not name.exists():
                M.atomic_write(name, data, sync=False)
            rel = name.relative_to(OUT).as_posix()
            row = {"id": uuid.uuid4().hex[:10], "file": rel, "kind": kind, "prompt": prompt, "width": w, "height": h, "seed": None,
                   "seconds": 0, "ts": int(time.time()), "parent": parent, "refs": [parent] if parent else [], "fav": False}
            with LOCK:
                existing = next((r for r in HIST if r["file"] == rel), None)
            if existing is None:
                add_rows([row])
                existing = row
            return self._send(200, existing)
        if path == "/api/export":
            b = self._json()
            return self._send(200, export_file(str(b.get("file", "")), str(b.get("format", "png")), as_int(b.get("quality"), "quality", -10**6, 10**6, 90),
                                               str(b.get("preset", "illustration")), as_int(b.get("max_side"), "max_side", -10**9, 10**9, 0), bool(b.get("lossless"))))
        if path == "/api/fav":
            body = self._json()
            with LOCK:
                hit = [r for r in HIST if r["file"] == body.get("file")]
                for r in hit:
                    r["fav"] = bool(body.get("fav"))
                if hit:
                    persist_history()
            return self._send(200, {"ok": True})
        if path == "/api/reveal":
            target = safe_path(str(self._json().get("file", "")))
            if not target.is_file():
                raise ApiError("not found", 404)
            reveal(target)
            return self._send(200, {"ok": True})
        if path == "/api/trash":
            rel = str(self._json().get("file", ""))
            target = safe_path(rel)
            if target.is_file():
                move_to_trash(target)
            with LOCK:
                HIST[:] = [r for r in HIST if r["file"] != rel]
                persist_history()
            return self._send(200, {"ok": True})
        if path == "/api/models/inspect":
            return self._hf(lambda: M.inspect_repo(str(self._json().get("url", "")), hf_token()))
        if path == "/api/models/add":
            body = self._json()
            return self._hf(lambda: models_add(body))
        if path == "/api/models/delete":
            return self._send(200, models_delete(str(self._json().get("id", ""))))
        if path == "/api/models/active":
            mid = str(self._json().get("id", ""))
            if REG.get(mid) is None:
                raise ValueError("unknown model")
            REG.set_active(mid)
            return self._send(200, public_models())
        if path == "/api/settings":
            b = self._json()
            unknown = sorted(str(k) for k in b if k != "hf_token")
            if unknown:
                raise ValueError("unknown setting: " + ", ".join(unknown))
            if "hf_token" in b:
                if not isinstance(b["hf_token"], str):
                    raise ValueError("hf_token must be a string")
                set_hf_token(b["hf_token"].strip())
            return self._send(200, {"token": bool(hf_token())})  # whether a token is set, never the token itself
        if path == "/api/chats":
            return self._send(200, chats_api(self._json()))
        m = re.fullmatch(r"/api/chats/([a-z0-9]+)(/delete)?", path)
        if m:
            return self._send(200, chats_api(self._json(), m.group(1), delete=bool(m.group(2))))
        if path == "/api/setup/start":
            body = self._json()
            ids = body.get("ids") or [c["id"] for c in components() if c["required"] or body.get("all")]
            if not isinstance(ids, list):
                raise ValueError("ids must be a list")
            return self._send(200, setup_start([str(i) for i in ids]))
        if path == "/api/setup/cancel":
            SETUP["cancel"] = True
            return self._send(200, {"ok": True})
        if path == "/api/quit":
            self._send(200, {"ok": True})
            threading.Thread(target=shutdown_server, daemon=True).start()
            return
        m = re.fullmatch(r"/api/cancel/([a-f0-9]+)", path)
        if m:
            job = JOBS.get(m.group(1))
            if not job:
                return self._send(404, {"error": "unknown job"})
            if job["status"] in ("queued", "running"):
                job["status"] = "cancelled"
                proc = CURRENT["proc"]
                if CURRENT["id"] == job["id"] and proc:
                    kill_tree(proc)
            return self._send(200, public(job))
        self._send(404, {"error": "not found"})


HTTPD = None


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64
    allow_reuse_address = SYSTEM != "Windows"  # on Windows SO_REUSEADDR would let two instances share one port

    def __init__(self, *args, **kwargs):
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(*args, **kwargs)

    def server_bind(self):  # skip socket.getfqdn(), which can stall for seconds on odd Windows networks
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[0], self.server_address[1]

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):  # too many open connections: refuse this one instead of piling up threads
            return self.shutdown_request(request)
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):  # resets and stalled clients are routine
            super().handle_error(request, client_address)


def shutdown_server(*_args) -> None:
    """Stop the running job (and its whole process tree), then the HTTP server."""
    SETUP["cancel"] = True
    proc = CURRENT.get("proc")
    if proc:
        kill_tree(proc, force=True)
    remove_server_json()
    if HTTPD is not None:
        threading.Thread(target=HTTPD.shutdown, daemon=True).start()


def remove_server_json() -> None:
    """Delete the port file, but only our own: a second instance that gave up must not erase the first one's."""
    f = DATA / "server.json"
    try:
        if json.loads(f.read_text(encoding="utf-8-sig")).get("pid") == os.getpid():
            f.unlink(missing_ok=True)
    except (OSError, ValueError, AttributeError):
        pass


def pid_alive(pid: int) -> bool:
    if SYSTEM == "Windows":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not h:
            return False
        alive = ctypes.windll.kernel32.WaitForSingleObject(h, 0) == 0x102  # WAIT_TIMEOUT: still running
        ctypes.windll.kernel32.CloseHandle(h)
        return alive
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def watch_parent(pid: int) -> None:
    """If the app that launched us disappears (crash, force quit), do not leave the server and the GPU job behind."""
    while True:
        time.sleep(2)
        if not pid_alive(pid):
            shutdown_server()
            return


def already_running(port: int) -> bool:
    """Is an ImageLoom server answering on `port`?"""
    try:
        with LOCAL_OPENER.open(urllib.request.Request(f"http://127.0.0.1:{port}/api/status", headers={"X-ImageLoom": "1"}), timeout=2) as r:
            return json.load(r).get("ok") is True
    except Exception:  # noqa: BLE001
        return False


def bind(preferred: int) -> Server:
    try:
        return Server(("127.0.0.1", preferred), Handler)
    except OSError as exc:
        if preferred and already_running(preferred):
            die(f"ImageLoom is already running on http://127.0.0.1:{preferred}  (open it with `imageloom ui`, stop it with `imageloom stop`)")
        M.warn(f"port {preferred} is busy ({exc.strerror or exc}); using a free one instead")
    try:
        return Server(("127.0.0.1", 0), Handler)
    except OSError as exc:
        die(f"cannot open a local port: {exc.strerror or exc}")


def startup_notes() -> list[str]:
    """What is missing, in plain words (the server still starts: the UI's setup screen can fix it)."""
    notes = []
    if not Path(sd_cli()).is_file():
        notes.append("engine: sd-cli was not found. The setup screen (or adding a model) downloads it; or install it and set IMAGELOOM_SD_CLI")
    if not any_model_ready():
        notes.append("models: none is ready yet. Open the setup screen for the default model, or run `imageloom models add <huggingface link>`")
    if Image is None:
        notes.append("Pillow is not installed: thumbnails, export and picture checks are limited (pip install pillow)")
    return notes


def main() -> None:
    global HTTPD
    for stream in (sys.stdout, sys.stderr):  # a non-cp1252 user name must not crash the startup banner
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    try:
        port = int(os.environ.get("IMAGELOOM_PORT") or CFG["port"])
    except ValueError:
        die(f"IMAGELOOM_PORT must be a port number, not {os.environ.get('IMAGELOOM_PORT')!r}")
    HTTPD = bind(port)  # first: a second instance must stop here, before it touches any state file
    CFG["port"] = HTTPD.server_address[1]
    M.atomic_write(DATA / "server.json", json.dumps({"port": CFG["port"], "pid": os.getpid(), "version": VERSION}), sync=False)
    tighten_token_file()
    load_history()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=backfill_alpha, daemon=True).start()
    for sig in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), shutdown_server)
    atexit.register(lambda: CURRENT.get("proc") and kill_tree(CURRENT["proc"], force=True))
    if os.environ.get("IMAGELOOM_PARENT_PID", "").isdigit():
        threading.Thread(target=watch_parent, args=(int(os.environ["IMAGELOOM_PARENT_PID"]),), daemon=True).start()
    print(f"ImageLoom {VERSION} listening on http://127.0.0.1:{CFG['port']}  (data: {DATA}, output: {OUT})", flush=True)
    for note in startup_notes():
        print("  " + note, flush=True)
    try:
        HTTPD.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        remove_server_json()


if __name__ == "__main__":
    main()
