#!/usr/bin/env python3
"""ImageLoom MCP server: lets any Model Context Protocol client (Claude Desktop, Claude Code, Cursor, VS Code, Zed, Cline, ...)
create, edit, upscale and export images, and manage models, with the local ImageLoom studio.

  imageloom mcp                      the same thing through the CLI
  python3 core/imageloom_mcp.py      stdio transport: newline-delimited JSON-RPC 2.0 on stdin/stdout, logs on stderr

Pure standard library (Python 3.9+). Its only network peer is the local ImageLoom server on 127.0.0.1, which it starts on demand exactly
like the CLI does: the HTTP plumbing, size and model rules are the CLI's own functions (core/imageloom is imported as a module).
Image jobs take minutes, so generation is asynchronous: generate_image returns a job id at once and wait_for_job / get_job collect the
result in slices of at most 50 s. See docs/MCP.md.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
from pathlib import Path

CORE = Path(__file__).resolve().parent
SERVER_NAME = "imageloom"
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")  # newest first
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603

MAX_LINE = 8 * 1024 * 1024  # one JSON-RPC message
MAX_PARALLEL_CALLS = 8
POLL_SECONDS = 1.0  # how often a waiting tool asks the ImageLoom server about its job
MAX_WAIT = 50  # a tool call never blocks longer than this, so clients with a 60 s tool timeout do not fail
CALL_BUDGET = 55  # submit + wait together
MAX_PREVIEWS = 4  # images attached to one result
PREVIEW_SIDE = 768
PREVIEW_BYTES = 400_000  # per preview (before base64): a result with four of them stays around two megabytes
MAX_PREVIEW_PIXELS = 100_000_000
MAX_INPUT_BYTES = 64 * 1024 * 1024  # the server refuses bigger uploads anyway
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif")
QUALITIES = ["fast", "standard", "hq", "max"]  # = the CLI's -q, in the order of its effort levels
TERMINAL = ("done", "error", "cancelled")
DEBUG = bool(os.environ.get("IMAGELOOM_MCP_DEBUG"))


def read_version() -> str:
    """The version of the ImageLoom core (VERSION in server.py), read as text: importing server.py would create folders."""
    try:
        m = re.search(r'^VERSION\s*=\s*"([^"]+)"', (CORE / "server.py").read_text(encoding="utf-8"), re.M)
        return m.group(1) if m else "0.0.0"
    except OSError:
        return "0.0.0"


VERSION = read_version()

INSTRUCTIONS = (
    "ImageLoom makes pictures on the user's own computer; it is slow (about 4 minutes for a 768 px image on a base Apple M4). "
    "generate_image, edit_image and upscale_image only submit a background job and return a job_id at once; then call wait_for_job "
    "(each call waits up to 50 s) until status is 'done' and the image paths arrive. Do not resubmit while a job runs. One job runs at a time. "
    "Start with quality 'fast'. add_model downloads many GB: it plans first (dry_run) and you must ask the user before dry_run=false."
)


def log(*parts) -> None:
    print("imageloom-mcp:", *parts, file=sys.stderr, flush=True)


def debug(*parts) -> None:
    if DEBUG:
        log(*parts)


def human_time(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s} s" if s < 90 else f"{s / 60:.1f} min" if s < 5400 else f"{s / 3600:.1f} h"


def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


class ToolError(Exception):
    """A failure the model can act on: becomes an `isError` tool result with this message (never a protocol error)."""


class SetupNeeded(ToolError):
    """The engine or every model is missing: only the user can fix that."""


class Cancelled(Exception):
    """The client cancelled the request (or closed stdin) while a tool was waiting."""


class RpcError(Exception):
    def __init__(self, code: int, message: str, data=None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


# ----------------------------------------------------------------------------------------- JSON Schema (the small subset we publish)
def prop(type_: str, description: str, **kw) -> dict:
    return {"type": type_, "description": description, **kw}


def _coerce(schema: dict, value):
    """Models often send numbers and booleans as strings: accept the ones that are unambiguous."""
    t = schema.get("type")
    if isinstance(value, bool):
        return value
    if t == "integer":
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"\s*[+-]?\d{1,18}\s*", value):
            return int(value)
    elif t == "number" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    elif t == "string" and "enum" in schema and isinstance(value, str):
        return next((e for e in schema["enum"] if e.lower() == value.strip().lower()), value)
    return value


def _check(schema: dict, value, path: str, errors: list):
    t = schema.get("type")
    if t == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
        value = value.strip().lower() == "true"
    value = _coerce(schema, value)
    if t == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: must be a string")
            return value
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: must be one of {', '.join(schema['enum'])} (got {value!r})")
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", 10**9):
            errors.append(f"{path}: length must be {schema.get('minLength', 0)}..{schema.get('maxLength', '')} characters (got {len(value)})")
        if "pattern" in schema and not re.search(schema["pattern"][:-1] + r"\Z" if schema["pattern"].endswith("$") else schema["pattern"], value):  # ($ would let a trailing newline through)
            errors.append(f"{path}: has the wrong format (got {value!r}). {schema.get('description', '')}".rstrip())
    elif t in ("integer", "number"):
        ok = isinstance(value, int) if t == "integer" else isinstance(value, (int, float))
        if not ok or isinstance(value, bool) or (isinstance(value, float) and value != value):
            errors.append(f"{path}: must be {'a whole number' if t == 'integer' else 'a number'} (got {value!r})")
            return value
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            errors.append(f"{path}: must be between {lo} and {hi} (got {value})")
        if "multipleOf" in schema and value % schema["multipleOf"]:
            errors.append(f"{path}: must be a multiple of {schema['multipleOf']} (got {value})")
    elif t == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: must be true or false")
    elif t == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: must be a list")
            return value
        if len(value) > schema.get("maxItems", 10**9):
            errors.append(f"{path}: at most {schema['maxItems']} items (got {len(value)})")
        value = [_check(schema.get("items", {}), v, f"{path}[{i}]", errors) for i, v in enumerate(value)]
    return value


def validate_args(name: str, schema: dict, args: dict) -> dict:
    """Check `args` against a tool's inputSchema; fills declared defaults. One ToolError lists every problem."""
    errors: list = []
    props = schema["properties"]
    out: dict = {}
    for k in args:
        if k not in props:
            errors.append(f"unknown argument '{k}' (allowed: {', '.join(props) or 'none'})")
    for k, s in props.items():
        if args.get(k) is not None:  # (null means "not given")
            out[k] = _check(s, args[k], k, errors)
        elif "default" in s:
            out[k] = s["default"]
    for k in schema.get("required", []):
        if args.get(k) is None:
            errors.append(f"missing required argument '{k}'")
    if errors:
        raise ToolError(f"invalid arguments for {name}:\n- " + "\n- ".join(errors))
    return out


# ----------------------------------------------------------------------------------------- results
class Result:
    def __init__(self, text: str, data: dict | None = None, images: list | None = None, is_error: bool = False):
        self.text, self.data, self.images, self.is_error = text, data, images or [], is_error

    def to_mcp(self, version: str) -> dict:
        out: dict = {"content": [{"type": "text", "text": self.text}, *self.images]}
        if self.data is not None and version >= "2025-06-18":  # (structuredContent exists from that revision on; the text carries everything for older clients)
            out["structuredContent"] = self.data
        out["isError"] = self.is_error
        return out


class Tool:
    def __init__(self, name, title, description, properties, required, hints, fn):
        self.name, self.title, self.description, self.fn = name, title, description, fn
        self.input_schema = {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}
        self.annotations = {"title": title, **hints}

    def describe(self) -> dict:
        return {"name": self.name, "title": self.title, "description": self.description, "inputSchema": self.input_schema, "annotations": self.annotations}


TOOLS: dict = {}


def tool(name: str, title: str, description: str, properties: dict, required=(), *, read_only: bool, destructive: bool, idempotent: bool, open_world: bool):
    def deco(fn):
        hints = {"readOnlyHint": read_only, "destructiveHint": destructive, "idempotentHint": idempotent, "openWorldHint": open_world}
        TOOLS[name] = Tool(name, title, description, properties, required, hints, fn)
        return fn
    return deco


class Ctx:
    """What a running tool call can do: notice cancellation, report progress."""

    def __init__(self, server: McpServer, request_id, token):
        self.server, self.request_id, self.token = server, request_id, token
        self.cancelled = threading.Event()  # the client sent notifications/cancelled for this request

    def check(self) -> None:
        if self.cancelled.is_set() or self.server.stopping.is_set():
            raise Cancelled

    def sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while True:
            self.check()
            left = end - time.monotonic()
            if left <= 0:
                return
            self.cancelled.wait(min(left, 0.25))

    def progress(self, value: float, total: float | None = None, message: str | None = None) -> None:
        if self.token is None:
            return
        params: dict = {"progressToken": self.token, "progress": value}
        if total is not None:
            params["total"] = total
        if message and self.server.version >= "2025-03-26":
            params["message"] = message
        self.server.notify("notifications/progress", params)


# ----------------------------------------------------------------------------------------- the ImageLoom side (through the CLI module)
def load_cli_module():
    """core/imageloom has no .py suffix: import it by path. Its module-level code only reads the environment and config files."""
    loader = importlib.machinery.SourceFileLoader("imageloom_cli", str(CORE / "imageloom"))
    spec = importlib.util.spec_from_loader("imageloom_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class Backend:
    """One place that talks to the local ImageLoom server, reusing the CLI's helpers (call, ensure_server, preflight, dims, prepare_ref ...)."""

    def __init__(self):
        self._cli = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._start_lock = threading.Lock()  # one server start at a time
        self._lib: tuple = (0.0, None)

    def set_cli(self, mod) -> None:
        self._cli = mod

    @property
    def cli(self):
        if self._cli is None:
            with self._load_lock:
                if self._cli is None and self._load_error is None:
                    try:
                        self._cli = load_cli_module()
                    except SystemExit as exc:  # (an unusable IMAGELOOM_PORT ends the CLI at import: tell the model, do not die)
                        self._load_error = str(exc.code or "the ImageLoom CLI module could not start")
                    except Exception as exc:  # noqa: BLE001
                        self._load_error = (f"the ImageLoom CLI module could not be loaded ({type(exc).__name__}: {exc}). This server reuses the `imageloom` CLI file that sits next to it in core/: "
                                            "run it from a full checkout of the repository.")
        if self._cli is None:
            raise ToolError(self._load_error or "the ImageLoom CLI module is unavailable")
        return self._cli

    # -- errors: CLI and network failures become messages the model can act on
    @contextlib.contextmanager
    def guard(self):
        cli = self.cli
        try:
            yield
        except cli.CliError as exc:
            raise ToolError(str(exc)) from None
        except (OSError, ValueError) as exc:  # URLError, refused connections, timeouts, an answer that is not JSON
            raise ToolError(self.describe_failure(exc)) from None

    def describe_failure(self, exc: Exception) -> str:
        base = self.cli.BASE
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason):
            return f"the ImageLoom server at {base} did not answer in time. It may be busy (for example loading a model); try again in a moment."
        if isinstance(exc, ValueError):
            return f"the server at {base} sent an answer that is not ImageLoom's. Is something else using that port?"
        if isinstance(exc, (urllib.error.URLError, ConnectionError)):
            return f"cannot reach the ImageLoom server at {base} ({reason}). If it was stopped, calling the tool again starts it."
        return f"{type(exc).__name__}: {exc.strerror or exc}" if isinstance(exc, OSError) else f"{type(exc).__name__}: {exc}"

    def api(self, path: str, body=None, *, timeout: float = 30, raw: bool = False) -> tuple:
        with self.guard():
            return self.cli.call(path, body, timeout=timeout, raw=raw)

    def ok(self, path: str, body=None, *, timeout: float = 30, what: str = "ImageLoom") -> dict:
        code, data = self.api(path, body, timeout=timeout)
        if code != 200 or not isinstance(data, dict):
            raise ToolError(f"{what} answered {code}: {data.get('error', data) if isinstance(data, dict) else data}")
        return data

    # -- the server: found or started
    def _sync_port(self) -> None:
        cli = self.cli
        if not cli.alive():  # the desktop app may have restarted on another port since we last looked
            try:
                cli.PORT = cli.discover_port()
            except SystemExit:
                pass
            cli.BASE = f"http://127.0.0.1:{cli.PORT}"

    def ensure(self) -> None:
        """Start the ImageLoom server when it is not running (the CLI's own logic)."""
        with self._start_lock:
            self._sync_port()
            with self.guard():
                self.cli.ensure_server()

    def connect(self) -> None:
        """For tools that only make sense on a running server (jobs live in its memory): never starts one."""
        self._sync_port()
        if not self.cli.alive():
            raise ToolError("ImageLoom is not running, so it has no jobs (they live in the server's memory and are lost when it quits). "
                            "Start a new one with generate_image, or look at finished pictures with recent_images.")

    def status(self) -> dict:
        return self.ok("/api/status", what="/api/status")

    def library(self) -> Path:
        """The folder the server saves pictures in (asked now and then: the app can be restarted with another one)."""
        stamp, lib = self._lib
        if lib is None or time.monotonic() - stamp > 30:
            out = self.status().get("output_dir")
            lib = Path(out).expanduser().resolve() if out else (Path.home() / "Pictures" / "ImageLoom").resolve()
            self._lib = (time.monotonic(), lib)
        return lib

    def preflight(self, need_models: bool = True) -> None:
        """Say plainly what is missing before any work is done (engine, models). The text is the CLI's, plus what an agent can do about it."""
        cli = self.cli
        try:
            with self.guard():
                try:
                    cli.preflight(need_models)
                except cli.CliError as exc:
                    raise SetupNeeded(str(exc)) from None
        except SetupNeeded as exc:
            raise ToolError(f"{exc}\nOnly the user can finish the setup (the ImageLoom app's setup screen downloads the engine and the default model, about 17 GB). "
                            "Tell them and stop retrying. add_model can also fetch the engine, but it downloads many GB, so ask the user first.") from None

    def pick_model(self, query: str | None, edit: bool) -> dict:
        with self.guard():
            model, why = self.cli.pick_model(query, edit)
        if model is None:
            raise ToolError(f"{why} (list_models shows what is installed)")
        return model


B = Backend()


# ----------------------------------------------------------------------------------------- images: input checks and previews
def sniff_image(head: bytes) -> str | None:
    """Which picture format the first bytes of a file announce (so a text file renamed to .png is not taken for one)."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:2] == b"BM":
        return "bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"):
        return "heic"
    return None


def image_arg(value, name: str) -> Path:
    """An input picture: must be an existing regular file, with an image extension AND an image header. Returns the resolved absolute path."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ToolError(f"{name} must be the path of an image file")
    raw = Path(os.path.expanduser(value.strip()))
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    try:
        path = raw.resolve(strict=True)
    except FileNotFoundError:
        raise ToolError(f"{name}: no such file: {raw}. Pass the absolute path of an existing image (relative paths are resolved against {Path.cwd()}).") from None
    except (OSError, RuntimeError) as exc:
        raise ToolError(f"{name}: cannot open {raw} ({exc})") from None
    if not path.is_file():
        raise ToolError(f"{name}: {path} is not a file")
    if path.suffix.lower() not in IMAGE_EXTS:
        raise ToolError(f"{name}: {path.name} is not an image file (accepted: {', '.join(e.lstrip('.') for e in IMAGE_EXTS)})")
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            head = fh.read(32)
    except OSError as exc:
        raise ToolError(f"{name}: cannot read {path} ({exc.strerror or exc})") from None
    if not 0 < size <= MAX_INPUT_BYTES:
        raise ToolError(f"{name}: {path.name} is {'empty' if size == 0 else gb(size) + ' (limit 64 MB)'}")
    if sniff_image(head) is None:
        raise ToolError(f"{name}: {path.name} does not contain a picture (its header is not PNG, JPEG, WebP, GIF, BMP, TIFF or HEIC)")
    return path


def png_header(data: bytes) -> tuple | None:
    """(width, height, has_alpha) of a PNG from its first 33 bytes."""
    if len(data) >= 33 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return w, h, data[25] in (4, 6)
    return None


def preview_backend() -> str | None:
    try:
        import PIL  # noqa: F401
        return "pillow"
    except ImportError:
        return "sips" if sys.platform == "darwin" and shutil.which("sips") else None


def make_preview(path: Path) -> tuple | None:
    """(mime, bytes) of a copy of a finished picture scaled to at most 768 px (JPEG, or PNG when it has transparency), or None.
    Pillow when installed, else macOS `sips`, else the file itself when it is a small PNG."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(33)
        size = path.stat().st_size
        png = png_header(head)
        if png and max(png[:2]) <= PREVIEW_SIDE and size <= PREVIEW_BYTES:
            return "image/png", path.read_bytes()
        if sniff_image(head) is None:
            return None  # (an SVG, say)
        try:
            from PIL import Image
        except ImportError:
            Image = None
        if Image is not None:
            with Image.open(path) as im:
                if im.width * im.height > MAX_PREVIEW_PIXELS:
                    return None
                alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
                if not alpha:
                    im.draft("RGB", (PREVIEW_SIDE * 2, PREVIEW_SIDE * 2))
                base = im.convert("RGBA" if alpha else "RGB")
            for side in (PREVIEW_SIDE, 640, 512, 384):  # (smaller only when the encoded picture is still too heavy)
                small = base.copy()
                small.thumbnail((side, side), Image.LANCZOS)
                for quality in (None,) if alpha else (85, 70):
                    buf = io.BytesIO()
                    if alpha:
                        small.save(buf, "PNG", optimize=True)
                    else:
                        small.save(buf, "JPEG", quality=quality, optimize=True)
                    if len(buf.getvalue()) <= PREVIEW_BYTES:
                        return ("image/png" if alpha else "image/jpeg"), buf.getvalue()
            return None
        if sys.platform == "darwin" and shutil.which("sips"):
            fmt, ext = ("png", "png") if png and png[2] else ("jpeg", "jpg")
            with tempfile.TemporaryDirectory() as td:
                dst = Path(td) / f"preview.{ext}"
                subprocess.run(["sips", "-Z", str(PREVIEW_SIDE), "-s", "format", fmt, "-s", "formatOptions", "80", str(path.resolve()), "--out", str(dst)],
                               capture_output=True, check=True, timeout=30)
                data = dst.read_bytes()
            return (f"image/{fmt}", data) if len(data) <= PREVIEW_BYTES else None
    except Exception as exc:  # noqa: BLE001 - a preview is a courtesy: the path is always in the text
        debug(f"no preview for {path.name}: {type(exc).__name__}: {exc}")
    return None


def preview_items(paths: list) -> list:
    out = []
    for p in paths[:MAX_PREVIEWS]:
        got = make_preview(p)
        if got:
            out.append({"type": "image", "data": base64.b64encode(got[1]).decode("ascii"), "mimeType": got[0]})
    return out


def in_library(lib: Path, name: str) -> Path:
    """A library file name from the server, as an absolute path that provably stays inside the library."""
    p = (lib / name).resolve()
    if lib not in p.parents:
        raise ToolError(f"the server named a file outside its library ({name}); refusing to use it")
    return p


def stage_upload(src: Path, lib: Path) -> str:
    """The server only reads pictures from its library: convert the file to PNG (sips or Pillow, like the CLI) and upload it. Returns its library name."""
    with B.guard(), tempfile.TemporaryDirectory() as td:
        png, _, _ = B.cli.prepare_ref(src, None, None, Path(td), 0)
        code, row = B.cli.call("/api/upload?name=" + urllib.parse.quote(src.name), png.read_bytes(), raw=True, timeout=CALL_BUDGET)
    if code != 200:
        raise ToolError(f"ImageLoom refused the picture: {row.get('error', row)}")
    return row["file"]


# ----------------------------------------------------------------------------------------- jobs
META: dict = {}  # job id -> what this process knows about a job it submitted (the server forgets the options of a chain)
META_LOCK = threading.Lock()


def remember(job_id: str, **info) -> None:
    with META_LOCK:
        META[job_id] = info
        while len(META) > 300:
            META.pop(next(iter(META)))


def fetch_raw(job_id: str) -> dict:
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{4,40}", job_id):
        raise ToolError(f"job_id must be the lowercase hex id returned by generate_image / edit_image / upscale_image / list_jobs (got {job_id!r})")
    code, job = B.api(f"/api/jobs/{job_id}")
    if code == 404:
        raise ToolError(f"no job with id {job_id}. ImageLoom forgets jobs when it restarts (and after 200 newer ones): list_jobs shows what it knows, recent_images shows finished pictures.")
    if code != 200 or not isinstance(job, dict):
        raise ToolError(f"ImageLoom answered {code} for job {job_id}: {job.get('error', job) if isinstance(job, dict) else job}")
    return job


def build_view(raw: dict) -> dict:
    """One status for the model: the server's job plus the auto-upscale jobs chained after it (upscale_to), seen as a single job."""
    lib = B.library()
    jid = raw["id"]
    meta = META.get(jid, {})
    status = raw.get("status", "error")
    own = [in_library(lib, f) for f in raw.get("images") or []]
    view: dict = {"job_id": jid, "mode": raw.get("mode"), "status": status, "phase": raw.get("phase"), "step": raw.get("step"), "steps": raw.get("steps"),
                  "image_index": raw.get("image_index"), "image_total": raw.get("image_total"), "prompt": raw.get("prompt"), "model": raw.get("model_name") or raw.get("model"),
                  "width": raw.get("width"), "height": raw.get("height"), "seeds": raw.get("seeds") or [], "error": raw.get("error"), "warnings": []}
    est = float(raw.get("eta") or 0) + float(meta.get("chain_eta") or 0)
    elapsed = float(raw.get("elapsed") or 0) if status == "running" else float(raw.get("seconds") or 0)
    images, originals, stage = own, [], "creating" if raw.get("mode") != "upscale" else "upscaling"
    if status == "done":
        ids = raw.get("next")
        waiting_for_chain = ids is None and bool(meta.get("upscale_to")) and time.monotonic() - meta.setdefault("seen_done", time.monotonic()) < 10  # (the server lists the chain a moment after "done")
        follow = []
        for nid in ids or []:
            try:
                follow.append(fetch_raw(nid))
            except ToolError:
                follow.append({"id": nid, "status": "error", "error": "the follow-up upscale job is unknown to the server"})
        live = [f for f in follow if f.get("status") in ("queued", "running")]
        if waiting_for_chain or live:
            run = next((f for f in live if f.get("status") == "running"), None)
            stage = "upscaling"
            elapsed += float((run or {}).get("elapsed") or 0)
            view.update(status="running", phase="upscaling" if run else "waiting for the upscale to start", step=(run or {}).get("step", 0), steps=(run or {}).get("steps", 0), error=None)
            view["intermediate_images"] = [str(p) for p in own]
            images = []
        elif follow:
            ups = [f for f in follow if f.get("status") == "done"]
            if ups:
                images, originals = [in_library(lib, n) for f in ups for n in f.get("images") or []], own
                view.update(width=ups[-1].get("width"), height=ups[-1].get("height"))
            for f in follow:
                elapsed += float(f.get("seconds") or 0)
                if f.get("status") != "done":
                    view["warnings"].append(f"the automatic upscale {f.get('status')}: {f.get('error') or 'no details'}; the un-upscaled pictures are returned")
        if raw.get("error") and view["status"] == "done":
            view["warnings"].append(str(raw["error"]))
            view["error"] = None
    view["stage"] = stage
    view["images"] = [{"path": str(p), "width": view["width"], "height": view["height"], **({"seed": view["seeds"][i]} if not originals and i < len(view["seeds"]) and raw.get("mode") != "upscale" else {})}
                      for i, p in enumerate(images)]
    if originals:
        view["originals"] = [str(p) for p in originals]
    if status == "queued":
        view["queue_position"] = raw.get("position")
    view["elapsed_seconds"] = round(elapsed, 1)
    if est:
        view["estimated_seconds"] = round(est)
        if view["status"] in ("queued", "running"):
            view["remaining_seconds"] = max(round(est - elapsed), 0)
    if raw.get("peak_gb"):
        view["peak_memory_gb"] = raw["peak_gb"]
    return view


def percent(view: dict) -> float:
    """0..99 while a job runs: steps when the engine reports them, otherwise phases. Never claims 100 before the end."""
    if view["status"] == "queued":
        return 0.0
    steps, step = view.get("steps") or 0, view.get("step") or 0
    phase = view.get("phase") or ""
    if view.get("stage") == "upscaling" and view["mode"] != "upscale":
        return min(80 + 19 * (step / steps if steps else 0), 99.0)
    scale = 0.8 if META.get(view["job_id"], {}).get("upscale_to") else 1.0
    if phase == "upscaling":
        p = 100 * step / steps if steps else 3
    elif phase == "denoising":
        n = view.get("image_total") or 1
        p = 8 + 87 * (max((view.get("image_index") or 1) - 1, 0) + (step / steps if steps else 0)) / n
    elif phase in ("decoding", "saving"):
        p = 96
    else:
        p = 3 if phase == "loading model" else 6
    return min(p * scale, 99.0)


def wait_job(ctx: Ctx, job_id: str, seconds: float) -> dict:
    """Poll until the job (and its chained upscales) ends or `seconds` pass; sends progress notifications when the client asked for them."""
    deadline = time.monotonic() + max(seconds, 0)
    last, last_msg = -1.0, ""
    while True:
        ctx.check()
        view = build_view(fetch_raw(job_id))
        if view["status"] in TERMINAL:
            return view
        msg = status_line(view)
        p = percent(view)
        if p > last or msg != last_msg:
            last = max(p, last + 0.1)
            ctx.progress(round(min(last, 99.9), 1), 100, msg)
            last_msg = msg
        left = deadline - time.monotonic()
        if left <= 0:
            return view
        ctx.sleep(min(POLL_SECONDS, left))


def status_line(v: dict) -> str:
    est = v.get("estimated_seconds")
    if v["status"] == "queued":
        return f"queued (position {v.get('queue_position') or 1})" + (f", needs about {human_time(est)} once it starts" if est else "")
    if v["status"] == "running":
        where = f"{v.get('phase') or 'working'}" + (f" {v.get('step')}/{v.get('steps')}" if v.get("steps") else "")
        tail = f", about {human_time(v['remaining_seconds'])} left (estimate)" if v.get("remaining_seconds") else (", taking longer than estimated" if est else "")
        return f"{where}, {human_time(v.get('elapsed_seconds', 0))} elapsed{tail}"
    return v["status"]


def render_job(v: dict, *, submitted: dict | None = None, timed_out: bool = False) -> Result:
    """Text (what a model reads) plus structuredContent, with image previews once the job is done."""
    jid, st = v["job_id"], v["status"]
    lines: list = []
    images: list = []
    data = dict(v)
    if submitted:
        data.update(submitted)
        lines.append(submitted["summary"])
    if st in ("queued", "running"):
        data["next_action"] = "wait_for_job"
        if timed_out:
            lines.append(f"Still {st} after waiting: {status_line(v)}.")
        elif not submitted:
            lines.append(f"Job {jid} is {st}: {status_line(v)}.")
        if v.get("intermediate_images"):
            lines.append("The first-stage picture is already saved (the upscale is still running): " + ", ".join(v["intermediate_images"]))
        lines.append(f'Call wait_for_job with job_id "{jid}" again (each call waits up to {MAX_WAIT} s) until status is done. Do not submit the same request again.')
    elif st == "done":
        data["next_action"] = None
        paths = [Path(i["path"]) for i in v["images"]]
        took = v["elapsed_seconds"]
        lines.append(f"Job {jid} is done in {human_time(took)}. Saved {len(paths)} image{'s' if len(paths) != 1 else ''} ({v.get('width')}x{v.get('height')}) in {paths[0].parent if paths else ''}:")
        lines += [f"- {i['path']}" + (f" (seed {i['seed']})" if "seed" in i else "") for i in v["images"]]
        if v.get("originals"):
            lines.append("Before the upscale: " + ", ".join(v["originals"]))
        lines += [f"Warning: {w}" for w in v["warnings"]]
        images = preview_items(paths)
        if images:
            lines.append(f"Preview{'s' if len(images) > 1 else ''} of the {'first ' + str(len(images)) if len(paths) > len(images) else 'finished'} image{'s' if len(images) > 1 else ''} attached (downscaled to at most {PREVIEW_SIDE} px); the files keep full resolution.")
        else:
            lines.append("No preview could be attached here; open the file to look at it.")
    elif st == "error":
        data["next_action"] = None
        lines.append(f"Job {jid} FAILED: {v.get('error') or 'unknown error'}. No image was produced. ImageLoom's log: <data folder>/logs/server.log. You may retry once with a smaller size or a different seed.")
    else:
        data["next_action"] = None
        lines.append(f"Job {jid} was cancelled.")
    data.pop("summary", None)
    return Result("\n".join(lines), data, images, is_error=(st == "error"))


# ----------------------------------------------------------------------------------------- submitting jobs
def queue_job(body: dict, meta: dict) -> dict:
    code, job = B.api("/api/generate", body, timeout=30)
    if code == 202 and isinstance(job, dict):
        remember(job["id"], **meta)
        return job
    msg = (job.get("error") if isinstance(job, dict) else None) or f"HTTP {code}"
    if code == 503:
        raise ToolError(f"ImageLoom cannot run jobs yet: {msg}\nOnly the user can finish the setup (ImageLoom app, setup screen, or `imageloom ui`). Tell them.")
    if code == 429:
        raise ToolError(f"{msg} Use list_jobs to see them, cancel_job to drop one, or wait.")
    raise ToolError(f"ImageLoom rejected the request: {msg}")


def finish_submit(ctx: Ctx, job: dict, info: dict, wait_seconds: int, started: float) -> Result:
    view = build_view(job)
    timed_out = False
    left = min(wait_seconds, CALL_BUDGET - (time.monotonic() - started))
    if wait_seconds > 0 and left > 0:
        view = wait_job(ctx, job["id"], left)
        timed_out = view["status"] in ("queued", "running")
    return render_job(view, submitted=info, timed_out=timed_out)


def queue_summary(job: dict, what: str, model: str, estimate: float, ahead: int, warnings: list) -> str:
    text = [f"Queued job {job['id']} with {model}: {what}. Estimated {human_time(estimate)} of work on this machine (an estimate from measured timings)"
            + (f"; {ahead} job{'s' if ahead != 1 else ''} ahead in ImageLoom's queue." if ahead else ".")]
    text += [f"Warning: {w}" for w in warnings]
    return " ".join(text[:1]) + ("\n" + "\n".join(text[1:]) if len(text) > 1 else "")


def estimate_for(w: int, h: int, count: int, edit: bool, model_id: str) -> float:
    code, est = B.api(f"/api/eta?w={w}&h={h}&n={count}&edit={1 if edit else 0}&model={urllib.parse.quote(model_id)}")
    return float(est.get("seconds", 0)) if code == 200 and isinstance(est, dict) else 0.0


def seed_arg(a: dict, body: dict) -> None:
    if a.get("seed") is not None:
        body["seed"] = a["seed"]


QUALITY_HELP = ("'fast' (default) is about 768 px, ~4 min on a base M4; 'standard' ~1024 px, ~7 min; 'hq' ~1536 px, ~30 min; 'max' ~2048 px, hours. "
                "Pick 'fast' unless the user asks for more; upscale_image makes a good draft big cheaply.")
WAIT_PROP = prop("integer", f"Optionally wait this many seconds (0-{MAX_WAIT}) for the job before returning; if it is not done by then you get the job id and keep waiting with wait_for_job. "
                 "Images take minutes, so 0 (return at once) is normal.", minimum=0, maximum=MAX_WAIT, default=0)
MODEL_PROP = prop("string", "Model id or part of its name, as listed by list_models. Omit to use the default model chosen in the ImageLoom app.", minLength=1, maxLength=120)
SEED_PROP = prop("integer", "Fixed seed for a reproducible result (omit for a random one).", minimum=0, maximum=2**53 - 1)
PATH_NOTE = "Pass an absolute path to an existing image file (png, jpg, webp, gif, bmp, tiff, heic)."


@tool("generate_image", "Generate image",
      "Create images from a text prompt with ImageLoom, a local AI image studio that runs entirely on this computer (no cloud). "
      "SLOW: one 768 px image takes about 4 minutes on a base Apple M4 (512 px ~2 min, 1024 px ~7 min), so this tool SUBMITS A BACKGROUND JOB and returns at once with "
      "a job_id and a time estimate; then call wait_for_job (each call waits up to 50 s) until status is 'done', which returns the saved PNG paths and a preview. "
      "Do not submit the same request again while it runs; one job runs at a time and others queue. "
      f"Quality: {QUALITY_HELP} Size: give width+height (multiples of 32, 512-2752) or aspect_ratio (1:1, 4:3, 16:9, 9:16 ...); quality sets the pixel budget. "
      "Write the prompt as a descriptive scene (subject, setting, light, style/medium, mood); put text that must appear in single quotes. "
      "transparent=true draws a real transparent background (logos, stickers; only models tagged 'alpha'). upscale_to chains an AI upscale after rendering (extra minutes). "
      "Pictures are saved in the ImageLoom output folder (usually ~/Pictures/ImageLoom). If the engine or a model is not installed the call fails with what the user must do.",
      {"prompt": prop("string", "What to draw, in plain language (up to 4000 characters). English and Russian work well.", minLength=1, maxLength=4000),
       "width": prop("integer", "Exact width in px (multiple of 32, 512-2752). Give together with height, or use aspect_ratio.", minimum=512, maximum=2752, multipleOf=32),
       "height": prop("integer", "Exact height in px (multiple of 32, 512-2752). Give together with width.", minimum=512, maximum=2752, multipleOf=32),
       "aspect_ratio": prop("string", "Shape instead of an exact size, e.g. '1:1' (default), '4:3', '3:4', '16:9', '9:16', '3:2'. The pixel count comes from quality.",
                            pattern=r"^[0-9]+(\.[0-9]+)?:[0-9]+(\.[0-9]+)?$"),
       "quality": prop("string", "Effort tier, as the CLI's -q. " + QUALITY_HELP, enum=QUALITIES, default="fast"),
       "model": MODEL_PROP, "seed": SEED_PROP,
       "count": prop("integer", "How many variants to render in one run (1-8). Each extra image costs about 30 s less than the first.", minimum=1, maximum=8, default=1),
       "transparent": prop("boolean", "Draw a real transparent background (only models with the 'alpha' capability, such as Qwen-Image 2.1; others ignore it).", default=False),
       "upscale_to": prop("string", "After rendering, AI-upscale the result to this target: 2x / 4x of the render, or 2k / 4k / 8k on the long side. Adds minutes.", enum=["2x", "4x", "2k", "4k", "8k"]),
       "upscale_model": prop("string", "Upscaler for upscale_to: quality (clean, default), fast (6x faster, drafts only: leaves a weave pattern) or illustration (drawings, anime).", enum=["quality", "fast", "illustration"], default="quality"),
       "wait_seconds": WAIT_PROP},
      ["prompt"], read_only=False, destructive=False, idempotent=False, open_world=False)
def generate_image(a: dict, ctx: Ctx) -> Result:
    started = time.monotonic()
    w, h, ratio = a.get("width"), a.get("height"), a.get("aspect_ratio")
    if (w is None) != (h is None):
        raise ToolError("give both width and height, or neither (aspect_ratio and quality then choose the size)")
    if w is not None and ratio:
        raise ToolError("give either width+height or aspect_ratio, not both")
    if w is None:
        with B.guard():
            w, h = B.cli.dims(ratio or "1:1", a["quality"])
    B.ensure()
    B.preflight()
    st = B.status()
    model = B.pick_model(a.get("model"), edit=False)
    count = a["count"]
    body: dict = {"mode": "generate", "prompt": a["prompt"], "width": w, "height": h, "count": count, "model": model["id"], "effort": QUALITIES.index(a["quality"])}
    seed_arg(a, body)
    warnings = []
    if a.get("transparent"):
        body["transparent"] = True
        if not (model.get("caps") or {}).get("alpha"):
            warnings.append(f"{model['name']} cannot draw transparent backgrounds; the option is ignored. Use a model tagged 'alpha' (list_models).")
    meta: dict = {}
    est = estimate_for(w, h, count, False, model["id"])
    if a.get("upscale_to"):
        body["then"] = {"upscale": {"model": a["upscale_model"], "target": a["upscale_to"]}}
        code, up = B.api(f"/api/eta?mode=upscale&w={w}&h={h}&model={a['upscale_model']}&target={a['upscale_to']}")
        if code != 200:
            raise ToolError(f"upscale_to {a['upscale_to']} is not possible: {up.get('error', up) if isinstance(up, dict) else up}")
        meta = {"upscale_to": a["upscale_to"], "chain_eta": up.get("seconds", 0)}
        est += meta["chain_eta"]
    if w * h > st.get("warn_pixels", 1_100_000):
        warnings.append("this size is heavy for a 16 GB Mac (it may swap); it is slow, and the user should close Xcode/Simulators.")
    if est > 1200:
        warnings.append(f"this will take about {human_time(est)}; a smaller quality followed by upscale_image is far cheaper.")
    ctx.check()  # (a cancelled request must not leave a job behind)
    job = queue_job(body, meta)
    ahead = (1 if st.get("running") else 0) + int(st.get("queued") or 0)
    what = f"{w}x{h} x{count}" + (f", then upscale to {a['upscale_to']}" if a.get("upscale_to") else "")
    info = {"summary": queue_summary(job, what, model["name"], est, ahead, warnings), "estimated_seconds": round(est), "jobs_ahead": ahead, "requested": {"width": w, "height": h, "count": count, "model": model["id"]}}
    return finish_submit(ctx, job, info, a["wait_seconds"], started)


@tool("edit_image", "Edit image",
      "Change an existing picture by describing the edit in plain words (for example 'replace the sky with a dramatic sunset, keep everything else the same'), "
      "using ImageLoom's local AI. The result follows the picture's aspect ratio but is rendered at a SMALL size (at least 512 px per side), because editing is much slower than creating: "
      "quality 'fast' (default) ~512 px (~2.5 min on a base M4), 'standard' ~640 px (~8 min), 'hq' ~768 px (~20 min), 'max' ~1024 px (hours). Use upscale_image afterwards for a bigger file. "
      "Needs an edit-capable model (tagged 'edit' in list_models; the default Qwen-Image 2.1 is). This SUBMITS A BACKGROUND JOB and returns a job_id; call wait_for_job (up to 50 s per call) until done. "
      "extra_references (max 2) are more pictures the instruction can refer to ('put the person from image 2 into image 1'). The original file is never modified; the result is a new file in the ImageLoom output folder. "
      "Fails with a clear message if the engine or an edit-capable model is missing.",
      {"image_path": prop("string", "The picture to edit. " + PATH_NOTE, minLength=1, maxLength=1024),
       "instruction": prop("string", "What to change, as a plain sentence; add 'keep everything else the same' for local edits.", minLength=1, maxLength=4000),
       "extra_references": prop("array", "Up to 2 more picture paths used as references by the instruction.", items={"type": "string", "minLength": 1, "maxLength": 1024}, maxItems=2),
       "quality": prop("string", "Effort tier; for editing it sets a smaller size than for creating (see the tool description).", enum=QUALITIES, default="fast"),
       "model": MODEL_PROP, "seed": SEED_PROP, "wait_seconds": WAIT_PROP},
      ["image_path", "instruction"], read_only=False, destructive=False, idempotent=False, open_world=False)
def edit_image(a: dict, ctx: Ctx) -> Result:
    started = time.monotonic()
    inputs = [image_arg(a["image_path"], "image_path")] + [image_arg(p, f"extra_references[{i}]") for i, p in enumerate(a.get("extra_references") or [])]
    B.ensure()
    B.preflight()
    st = B.status()
    model = B.pick_model(a.get("model"), edit=True)
    ctx.check()
    with B.guard(), tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, fw, fh = B.cli.prepare_ref(inputs[0], None, None, tmp, 0)
        w, h = B.cli.dims(f"{fw}:{fh}", a["quality"], edit=True)  # keep the picture's aspect ratio at the edit tier
        refs = []
        for i, src in enumerate(inputs):
            ctx.check()
            path, _, _ = B.cli.prepare_ref(src, w, h, tmp, i)
            code, row = B.cli.call("/api/upload?name=" + urllib.parse.quote(src.name), path.read_bytes(), raw=True, timeout=CALL_BUDGET)
            if code != 200:
                raise ToolError(f"ImageLoom refused {src.name}: {row.get('error', row)}")
            refs.append(row["file"])
    body = {"mode": "edit", "prompt": a["instruction"], "width": w, "height": h, "count": 1, "refs": refs, "model": model["id"], "effort": QUALITIES.index(a["quality"])}
    seed_arg(a, body)
    est = estimate_for(w, h, 1, True, model["id"])
    warnings = []
    if w * h > st.get("warn_pixels_edit", 420_000):
        warnings.append("editing at this size is very slow and memory-heavy on a 16 GB Mac.")
    ctx.check()
    job = queue_job(body, {})
    ahead = (1 if st.get("running") else 0) + int(st.get("queued") or 0)
    info = {"summary": queue_summary(job, f"edit at {w}x{h} ({len(inputs)} picture{'s' if len(inputs) > 1 else ''})", model["name"], est, ahead, warnings),
            "estimated_seconds": round(est), "jobs_ahead": ahead, "requested": {"width": w, "height": h, "model": model["id"]}}
    return finish_submit(ctx, job, info, a["wait_seconds"], started)


@tool("upscale_image", "Upscale image",
      "Enlarge a picture with an AI upscaler (Real-ESRGAN family) running locally: much cheaper than rendering large (a 512 px draft plus a 4K upscale beats a native 1536 px render). "
      "target: '2x' or '4x' of the input, or '2k' / '4k' / '8k' = 2560 / 3840 / 7680 px on the long side (limit 8192 px). "
      "model: 'quality' (default; Real-ESRGAN x4plus, clean, about 4 min per megapixel of INPUT), 'fast' (6x faster but leaves a visible weave pattern on skies and smooth areas: drafts only), "
      "'illustration' (drawings, posters, anime). It sharpens but cannot invent true detail, so judge the composition at small size first and upscale only the keeper. "
      "This SUBMITS A BACKGROUND JOB and returns a job_id plus an estimate; call wait_for_job (up to 50 s per call) until done. The original is untouched; the result is a new PNG in the ImageLoom output folder (a 4K PNG is 20-40 MB). "
      "Fails with a clear message if the engine or the chosen upscaler weights are missing.",
      {"image_path": prop("string", "The picture to enlarge. " + PATH_NOTE, minLength=1, maxLength=1024),
       "target": prop("string", "Result size: 2x, 4x (multiples of the input) or 2k, 4k, 8k (long side 2560 / 3840 / 7680 px).", enum=["2x", "4x", "2k", "4k", "8k"]),
       "model": prop("string", "Upscaler: quality (default), fast (drafts) or illustration.", enum=["quality", "fast", "illustration"], default="quality"),
       "wait_seconds": WAIT_PROP},
      ["image_path", "target"], read_only=False, destructive=False, idempotent=False, open_world=False)
def upscale_image(a: dict, ctx: Ctx) -> Result:
    started = time.monotonic()
    src = image_arg(a["image_path"], "image_path")
    B.ensure()
    B.preflight(need_models=False)
    st = B.status()
    model, target = a["model"], a["target"]
    if not (st.get("upscale") or {}).get(model, {}).get("ready", True):
        ready = [k for k, v in (st.get("upscale") or {}).items() if v.get("ready")]
        raise ToolError(f"the '{model}' upscaler weights are not installed" + (f"; installed: {', '.join(ready)}." if ready else ". Run `imageloom doctor` or the app's setup screen to download them (the user must do this)."))
    lib = B.library()
    ctx.check()
    rel = src.relative_to(lib).as_posix() if lib in src.parents and src.suffix.lower() == ".png" else stage_upload(src, lib)
    with B.guard():
        sw, sh = B.cli.png_size(lib / rel)
    code, est = B.api(f"/api/eta?mode=upscale&w={sw}&h={sh}&model={model}&target={target}")
    if code != 200:
        raise ToolError(f"cannot upscale to {target}: {est.get('error', est) if isinstance(est, dict) else est}")
    ctx.check()
    job = queue_job({"mode": "upscale", "src": rel, "model": model, "target": target}, {})
    ahead = (1 if st.get("running") else 0) + int(st.get("queued") or 0)
    info = {"summary": queue_summary(job, f"upscale {sw}x{sh} -> {est['out_w']}x{est['out_h']} ({model})", "the " + model + " upscaler", est["seconds"], ahead, []),
            "estimated_seconds": round(est["seconds"]), "jobs_ahead": ahead, "requested": {"source": f"{sw}x{sh}", "result": f"{est['out_w']}x{est['out_h']}", "model": model}}
    return finish_submit(ctx, job, info, a["wait_seconds"], started)


@tool("export_image", "Export image",
      "Convert a picture to PNG, WebP (lossy or lossless), JPEG, or trace it into an SVG vector, with ImageLoom. Runs synchronously and returns the new file (usually within seconds; SVG tracing of a large image can take up to a minute). "
      "Transparency is kept for PNG, WebP and SVG (JPEG is flattened on white). SVG is a tracing of the raster: great for logos, icons and flat illustrations (use preset 'logo'), noisy and large for photos. "
      "quality (50-100) applies to WebP/JPEG; lossless to WebP; preset to SVG: logo (few colours, clean shapes), illustration, detailed, bw (one-colour silhouette); max_side shrinks the long side (SVG default 1536). "
      "The original is never touched: the result is a new file under the ImageLoom output folder (_exports/), and its absolute path is returned.",
      {"image_path": prop("string", "The picture to convert. " + PATH_NOTE, minLength=1, maxLength=1024),
       "format": prop("string", "Output format.", enum=["png", "webp", "jpeg", "svg"]),
       "quality": prop("integer", "WebP/JPEG quality 50-100 (default 90).", minimum=50, maximum=100, default=90),
       "lossless": prop("boolean", "WebP only: lossless compression.", default=False),
       "preset": prop("string", "SVG tracing preset (default logo).", enum=["logo", "illustration", "detailed", "bw"], default="logo"),
       "max_side": prop("integer", "Shrink so the long side is at most this many px (0 = keep; SVG tracing uses 1536 unless set).", minimum=0, maximum=16384, default=0)},
      ["image_path", "format"], read_only=False, destructive=False, idempotent=False, open_world=False)
def export_image(a: dict, ctx: Ctx) -> Result:
    src = image_arg(a["image_path"], "image_path")
    B.ensure()
    lib = B.library()
    ctx.check()
    rel = src.relative_to(lib).as_posix() if lib in src.parents else stage_upload(src, lib)
    code, r = B.api("/api/export", {"file": rel, "format": a["format"], "quality": a["quality"], "preset": a["preset"], "lossless": a["lossless"], "max_side": a["max_side"]}, timeout=CALL_BUDGET)
    if code != 200 or not isinstance(r, dict):
        raise ToolError(f"export failed: {r.get('error', r) if isinstance(r, dict) else r}")
    path = in_library(lib, r["file"])
    data = {"path": str(path), "format": r.get("format"), "bytes": r.get("bytes"), "width": r.get("width"), "height": r.get("height"), "alpha": r.get("alpha"), "source": str(src)}
    if "paths" in r:
        data["svg_paths"] = r["paths"]
    lines = [f"Exported {a['format'].upper()} ({r.get('width')}x{r.get('height')}, {(r.get('bytes') or 0) / 1e6:.2f} MB" + (", transparent" if r.get("alpha") else "") + f"): {path}"]
    if "paths" in r:
        lines.append(f"The SVG has {r['paths']} vector paths.")
    images = preview_items([path]) if a["format"] != "svg" else []
    return Result("\n".join(lines), data, images)


@tool("get_job", "Get job status",
      "Look up an ImageLoom job started by generate_image, edit_image or upscale_image: status (queued, running, done, error, cancelled), phase, step/steps, elapsed time, "
      "an estimate of the time left, and when done the absolute paths of the finished images plus a preview. Returns immediately (use wait_for_job to block). "
      "A job that was asked to upscale_to is reported as one job: it stays 'running' until the upscaled pictures exist. Jobs live in the server's memory and are forgotten when ImageLoom restarts.",
      {"job_id": prop("string", "The job_id returned when the job was submitted (lowercase hex such as 3f9c1a2b7e40).", pattern=r"^[a-f0-9]{4,40}$")},
      ["job_id"], read_only=True, destructive=False, idempotent=True, open_world=False)
def get_job(a: dict, ctx: Ctx) -> Result:
    B.connect()
    return render_job(build_view(fetch_raw(a["job_id"])))


@tool("wait_for_job", "Wait for job",
      f"Block until an ImageLoom job finishes or timeout_seconds pass (default 30, at most {MAX_WAIT}, so the call stays inside a client's tool timeout). "
      "Image jobs take minutes: call this repeatedly with the same job_id until status is 'done' (it returns the image paths and a preview) or 'error'. "
      "While waiting it sends progress notifications if the client supplied a progress token. If the time runs out first, the result shows the current status and tells you to call again; the job keeps running either way.",
      {"job_id": prop("string", "The job_id returned when the job was submitted (lowercase hex such as 3f9c1a2b7e40).", pattern=r"^[a-f0-9]{4,40}$"),
       "timeout_seconds": prop("integer", f"How long to wait in this call (1-{MAX_WAIT}).", minimum=1, maximum=MAX_WAIT, default=30)},
      ["job_id"], read_only=True, destructive=False, idempotent=True, open_world=False)
def wait_for_job(a: dict, ctx: Ctx) -> Result:
    B.connect()
    view = wait_job(ctx, a["job_id"], a["timeout_seconds"])
    return render_job(view, timed_out=view["status"] in ("queued", "running"))


@tool("cancel_job", "Cancel job",
      "Cancel a queued or running ImageLoom job (and the automatic upscale chained after it). The work done so far is lost and the engine process is stopped; a job that already finished is left alone. "
      "The queue is shared with the ImageLoom app, so only cancel jobs you submitted (or that the user asked you to stop).",
      {"job_id": prop("string", "The job_id to cancel (lowercase hex such as 3f9c1a2b7e40).", pattern=r"^[a-f0-9]{4,40}$")},
      ["job_id"], read_only=False, destructive=True, idempotent=True, open_world=False)
def cancel_job(a: dict, ctx: Ctx) -> Result:
    B.connect()
    raw = fetch_raw(a["job_id"])
    targets = [raw]
    for nid in raw.get("next") or []:
        with contextlib.suppress(ToolError):
            targets.append(fetch_raw(nid))
    cancelled = []
    for job in targets:
        if job.get("status") in ("queued", "running"):
            code, r = B.api(f"/api/cancel/{job['id']}", {})
            if code != 200:
                raise ToolError(f"ImageLoom could not cancel job {job['id']}: {r.get('error', r) if isinstance(r, dict) else r}")
            cancelled.append(job["id"])
    if not cancelled:
        return Result(f"Job {raw['id']} is already {raw.get('status')}; nothing to cancel.", {"job_id": raw["id"], "status": raw.get("status"), "cancelled": []})
    return Result(f"Cancelled job{'s' if len(cancelled) > 1 else ''} {', '.join(cancelled)}. The queue moves on to the next job.", {"job_id": raw["id"], "status": "cancelled", "cancelled": cancelled})


@tool("list_jobs", "List jobs",
      "List the jobs the running ImageLoom server remembers (the last 25, newest first): id, status, kind, size, prompt. Includes jobs started from the ImageLoom app and the command line, "
      "because they share one queue. Use get_job for details of one job; recent_images lists finished pictures even after a restart.",
      {}, read_only=True, destructive=False, idempotent=True, open_world=False)
def list_jobs(a: dict, ctx: Ctx) -> Result:
    try:
        B.connect()
    except ToolError:
        return Result("ImageLoom is not running, so there are no jobs.", {"jobs": []})
    code, rows = B.api("/api/jobs")
    if code != 200 or not isinstance(rows, list):
        raise ToolError(f"ImageLoom answered {code} for /api/jobs")
    jobs = []
    for r in reversed(rows):
        jobs.append({"job_id": r.get("id"), "status": r.get("status"), "mode": r.get("mode"), "width": r.get("width"), "height": r.get("height"), "prompt": (r.get("prompt") or "")[:120],
                     "model": r.get("model_name") or r.get("model"), "step": r.get("step"), "steps": r.get("steps"), "phase": r.get("phase"), "estimated_seconds": r.get("eta"),
                     "images": len(r.get("images") or []), "queue_position": r.get("position")})
    if not jobs:
        return Result("ImageLoom has no jobs in memory.", {"jobs": []})
    lines = ["Jobs known to ImageLoom (newest first; use get_job for details):"]
    for j in jobs:
        prog = f" {j['phase']} {j['step']}/{j['steps']}" if j["status"] == "running" and j.get("steps") else ""
        lines.append(f"- {j['job_id']}  {j['status']}{prog}  {j['mode']} {j['width']}x{j['height']}  \"{j['prompt']}\"")
    return Result("\n".join(lines), {"jobs": jobs})


# ----------------------------------------------------------------------------------------- models and the gallery
def model_row(m: dict, active: str | None) -> dict:
    status = m.get("status", "ready")
    if status == "ready" and not m.get("ready"):
        status = "files_missing"  # (the built-in entry says "ready" even before its download)
    row = {"id": m["id"], "name": m["name"], "status": status, "ready": bool(m.get("ready")), "size_bytes": m.get("size", 0), "family": m.get("family_label") or m.get("family"),
           "edit": bool(m.get("can_edit")), "alpha": bool((m.get("caps") or {}).get("alpha")), "experimental": bool(m.get("experimental")), "verified": bool(m.get("verified")),
           "default": m["id"] == active, "license": m.get("license") or ""}
    dl = (m.get("downloading") or {}).values()
    total, got = sum(x.get("total", 0) for x in dl), sum(x.get("bytes", 0) for x in dl)
    if status == "downloading" and total:
        row["download_progress"] = round(got / total, 3)
    return row


@tool("list_models", "List models",
      "List the image models installed in ImageLoom (the server is started if needed): id, name, status (ready, downloading, error, files_missing), size, whether it can edit photos ('edit') or draw "
      "transparent backgrounds ('alpha'), whether its recipe is experimental (untested), and which one is the default. Pass an id (or part of the name) as `model` to generate_image / edit_image. "
      "Also shows the free disk space. Downloading models show their progress here.",
      {}, read_only=True, destructive=False, idempotent=True, open_world=False)
def list_models(a: dict, ctx: Ctx) -> Result:
    B.ensure()
    d = B.ok("/api/models", what="/api/models")
    rows = [model_row(m, d.get("active")) for m in d.get("models", [])]
    lines = ["Models (* = default):"]
    for r in rows:
        flags = ", ".join(x for x, on in (("edit", r["edit"]), ("alpha", r["alpha"]), ("experimental", r["experimental"])) if on)
        state = r["status"] + (f" {r['download_progress'] * 100:.0f}%" if "download_progress" in r else "")
        lines.append(f"{'*' if r['default'] else ' '} {r['id']} | {r['name']} | {state} | {gb(r['size_bytes'])}" + (f" | {flags}" if flags else ""))
    lines.append(f"Free disk space: {gb(d.get('free_bytes', 0))}. 'edit' = can edit photos, 'alpha' = can draw transparent backgrounds, 'experimental' = recipe not verified by the author.")
    if not any(r["ready"] for r in rows):
        lines.append("No model is ready yet: the user has to finish ImageLoom's setup (setup screen), or add_model can download one (ask the user first: it is many GB).")
    return Result("\n".join(lines), {"models": rows, "default_model": d.get("active"), "free_disk_bytes": d.get("free_bytes", 0)})


@tool("add_model", "Add model from Hugging Face",
      "Plan or perform the download of a new image model from a Hugging Face repository link (https://huggingface.co/owner/repo, or owner/repo). "
      "Default is dry_run=true: nothing is downloaded; you get the plan (detected family, the weights file chosen with its size, the companion files, the sd-cli engine if missing, the total size, free disk space, warnings, other file choices). "
      "THESE DOWNLOADS ARE MANY GIGABYTES (typically 4-20 GB): show the plan to the user and ASK for their explicit OK BEFORE calling again with dry_run=false; never download on your own initiative. "
      "With dry_run=false the download starts in the background (resumable) and the tool returns at once; follow it with list_models (status 'downloading' becomes 'ready'). "
      "Only one download batch runs at a time. Only Qwen-Image 2.1 Turbo is verified by the author; other families are experimental. Gated repositories need a Hugging Face token that the user adds in the ImageLoom app. "
      "Talks to huggingface.co (and, for the engine, github.com) through the ImageLoom server.",
      {"url": prop("string", "Hugging Face link such as https://huggingface.co/leejet/Z-Image-Turbo-GGUF, or owner/repo.", minLength=3, maxLength=500),
       "file": prop("string", "Weights file to use (path or file name from the plan's choices). Default: the best one that fits a 16 GB Mac.", minLength=1, maxLength=300),
       "family": prop("string", "Override the detected family (qwen-image-2.1, z-image, flux1, sdxl, sd15, custom).", maxLength=40),
       "variant": prop("string", "turbo / base / schnell / dev (default: detected).", maxLength=20),
       "dry_run": prop("boolean", "true (default) = only report the plan; false = start the download (ask the user first).", default=True)},
      ["url"], read_only=False, destructive=False, idempotent=True, open_world=True)
def add_model(a: dict, ctx: Ctx) -> Result:
    B.ensure()
    code, pl = B.api("/api/models/inspect", {"url": a["url"]}, timeout=CALL_BUDGET - 5)
    if code != 200 or not isinstance(pl, dict):
        msg = pl.get("error", pl) if isinstance(pl, dict) else pl
        if isinstance(pl, dict) and pl.get("need_token"):
            raise ToolError(f"{msg}\nThis repository is gated. The user must accept its terms on huggingface.co and paste a Hugging Face token in the ImageLoom app (Add model dialog); then retry.")
        raise ToolError(f"cannot inspect that repository: {msg}")
    cands = pl.get("candidates") or []
    want = a.get("file") or pl.get("default")
    cand = next((c for c in cands if c["path"] == want or c["path"].split("/")[-1] == want), None)
    if cand is None:
        raise ToolError(("choose `file` from: " + ", ".join(f"{c['path']} ({gb(c['size'])})" for c in cands[:20])) if cands else "that repository has no usable weights file (.gguf or a single .safetensors); " + "; ".join(pl.get("warnings") or []))
    family = a.get("family") or pl["family"]
    if family not in (pl.get("families") or {family: 1}):
        raise ToolError(f"unknown family '{family}'; use one of: {', '.join(pl['families'])}")
    st = B.status()
    free = B.ok("/api/models", what="/api/models").get("free_bytes", 0)
    engine_bytes = 0 if st.get("engine_found") else 35_000_000
    comps = [{"label": c["label"], "file": c["file"], "size_bytes": c["size"], "optional": bool(c.get("optional"))} for c in pl.get("companions") or []]
    total = cand["size"] + sum(c["size_bytes"] for c in comps) + engine_bytes
    plan = {"dry_run": a["dry_run"], "repo": pl["repo"], "name": pl["name"], "family": family, "family_label": (pl.get("families") or {}).get(family, family), "variant": a.get("variant") or pl["variant"],
            "license": pl.get("license") or "", "gated": bool(pl.get("gated")), "experimental": family != "qwen-image-2.1",
            "weights": {"file": cand["path"], "size_bytes": cand["size"], "quant": cand.get("quant") or ""}, "companions": comps, "engine_download_bytes": engine_bytes,
            "total_download_bytes_max": total, "free_disk_bytes": free, "fits_on_disk": total + 2_000_000_000 <= free, "warnings": pl.get("warnings") or [],
            "other_files": [{"file": c["path"], "size_bytes": c["size"], "quant": c.get("quant") or ""} for c in cands[:12] if c is not cand]}
    lines = [f"{pl['name']}: family {plan['family_label']}{' (EXPERIMENTAL: the recipe is untested)' if plan['experimental'] else ''}, license {plan['license'] or 'unknown'}.",
             f"Weights: {cand['path']} ({gb(cand['size'])})"] + [f"+ {c['label']} ({gb(c['size_bytes'])})" + (" (optional)" if c["optional"] else "") for c in comps]
    if engine_bytes:
        lines.append("+ the sd-cli engine (~35 MB; not installed yet, fetched first)")
    lines.append(f"Total download: up to {gb(total)} (files already on disk are skipped). Free disk: {gb(free)}" + ("" if plan["fits_on_disk"] else "  -- NOT ENOUGH SPACE (a 2 GB reserve is kept)") + ".")
    lines += [f"Warning: {w}" for w in plan["warnings"]]
    if a["dry_run"]:
        lines.append("Nothing was downloaded (dry run). This is a large download: ask the user to confirm, then call add_model again with dry_run=false.")
        if plan["other_files"]:
            lines.append("Other weight files: " + "; ".join(f"{o['file']} ({gb(o['size_bytes'])})" for o in plan["other_files"][:8]))
        return Result("\n".join(lines), plan)
    ctx.check()
    code, r = B.api("/api/models/add", {"repo": pl["repo"], "family": family, "variant": a.get("variant") or pl["variant"], "file": cand["path"], "name": pl["name"]}, timeout=CALL_BUDGET - 5)
    if code != 200 or not isinstance(r, dict):
        msg = r.get("error", r) if isinstance(r, dict) else r
        raise ToolError(f"the download did not start: {msg}")
    mid = next((m["id"] for m in r.get("models", []) if m.get("status") == "downloading"), None)
    plan["model_id"] = mid
    lines.append(f"Download started for model '{mid}' and continues in the background (it resumes if interrupted). Check list_models until its status is 'ready'; it cannot be used before.")
    return Result("\n".join(lines), plan)


@tool("set_default_model", "Set default model",
      "Make an installed model the default that generate_image and edit_image use when no `model` is given. This is the same setting as the model picker in the ImageLoom app, so the app changes too. "
      "Take the id from list_models. Changing it back is just another call.",
      {"id": prop("string", "Exact model id from list_models.", minLength=1, maxLength=120)},
      ["id"], read_only=False, destructive=False, idempotent=True, open_world=False)
def set_default_model(a: dict, ctx: Ctx) -> Result:
    B.ensure()
    code, r = B.api("/api/models/active", {"id": a["id"]})
    if code == 400:
        ids = [m["id"] for m in B.ok("/api/models", what="/api/models").get("models", [])]
        raise ToolError(f"unknown model id '{a['id']}'. Installed ids: {', '.join(ids)}")
    if code != 200 or not isinstance(r, dict):
        raise ToolError(f"ImageLoom answered {code}: {r.get('error', r) if isinstance(r, dict) else r}")
    row = next((model_row(m, r.get("active")) for m in r.get("models", []) if m["id"] == a["id"]), None)
    note = "" if not row or row["ready"] else f" Note: it is not ready yet (status {row['status']}), so jobs will fail until it is."
    return Result(f"The default model is now '{a['id']}'.{note}", {"default_model": r.get("active"), "model": row})


@tool("recent_images", "Recent images",
      "List the newest pictures in ImageLoom's gallery (created, edited, upscaled or composited; newest first) with absolute file paths, prompt, size, seed, model and time. "
      "Works after restarts, unlike job ids. Set previews (1-4) to attach downscaled previews of the newest ones so you can look at them.",
      {"limit": prop("integer", "How many items (1-50, default 10).", minimum=1, maximum=50, default=10),
       "previews": prop("integer", "Attach previews of this many of the newest items (0-4, default 0).", minimum=0, maximum=MAX_PREVIEWS, default=0)},
      read_only=True, destructive=False, idempotent=True, open_world=False)
def recent_images(a: dict, ctx: Ctx) -> Result:
    B.ensure()
    code, rows = B.api("/api/history")
    if code != 200 or not isinstance(rows, list):
        raise ToolError(f"ImageLoom answered {code} for /api/history")
    lib = B.library()
    items = []
    for r in rows:
        if r.get("kind") == "upload" or not isinstance(r.get("file"), str):
            continue
        try:
            path = in_library(lib, r["file"])
        except ToolError:
            continue
        items.append({"path": str(path), "kind": r.get("kind"), "prompt": r.get("prompt") or "", "width": r.get("width"), "height": r.get("height"), "seed": r.get("seed"), "model": r.get("model"),
                      "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])) if isinstance(r.get("ts"), (int, float)) else None, "favorite": bool(r.get("fav")),
                      "transparent": bool(r.get("transparent") or r.get("alpha"))})
        if len(items) >= a["limit"]:
            break
    if not items:
        return Result("The gallery is empty: nothing has been created yet.", {"images": []})
    lines = [f"The {len(items)} newest gallery item{'s' if len(items) != 1 else ''} (newest first):"]
    for n, i in enumerate(items, 1):
        lines.append(f"{n}. {i['path']}  [{i['kind']}, {i['width']}x{i['height']}" + (f", seed {i['seed']}" if i["seed"] is not None else "") + f", {i['created'] or '?'}]" + (f" \"{i['prompt'][:140]}\"" if i["prompt"] else ""))
    images = preview_items([Path(i["path"]) for i in items[:a["previews"]]]) if a["previews"] else []
    if images:
        lines.append(f"Previews of the newest {len(images)} attached (downscaled to at most {PREVIEW_SIDE} px).")
    return Result("\n".join(lines), {"images": items}, images)


# ----------------------------------------------------------------------------------------- the protocol
class McpServer:
    """JSON-RPC 2.0 over newline-delimited streams. Quick requests are answered on the reading thread; tools/call runs on its own thread so
    ping, cancellation notifications and further calls are handled while a tool waits for a job."""

    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.version = PROTOCOL_VERSIONS[0]
        self.init_seen = False  # (we do not insist on the handshake order: requests before `initialize` are answered too)
        self.client: dict = {}
        self.inflight: dict = {}
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.stopping = threading.Event()
        self.threads: set = set()
        self.slots = threading.BoundedSemaphore(MAX_PARALLEL_CALLS)

    # -- output
    def send(self, obj) -> None:
        data = json.dumps(obj, ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n"  # (ASCII-only: no U+2028 or friends for line-based readers)
        with self.write_lock:
            try:
                self.writer.write(data)
                self.writer.flush()
            except (BrokenPipeError, ValueError, OSError):
                self.stopping.set()  # the client is gone

    def notify(self, method: str, params: dict) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def fail(self, rid, code: int, message: str, data=None) -> dict:
        err: dict = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": rid, "error": err}

    # -- input
    def run(self) -> int:
        try:
            while not self.stopping.is_set():
                line = self.read_line()
                if line is None:
                    break  # EOF: the client closed our stdin
                if line is False:
                    self.send(self.fail(None, PARSE_ERROR, f"Parse error: a message is limited to {MAX_LINE // 2**20} MB"))
                    continue
                self.on_line(line)
        except KeyboardInterrupt:
            pass
        self.shutdown()
        return 0

    def read_line(self):
        data = self.reader.readline(MAX_LINE + 1)
        if not data:
            return None
        if len(data) > MAX_LINE and not data.endswith(b"\n"):
            while True:  # skip the rest of an oversized line
                chunk = self.reader.readline(MAX_LINE)
                if not chunk or chunk.endswith(b"\n"):
                    break
            return False
        return data

    def on_line(self, raw: bytes) -> None:
        raw = raw.strip()
        if not raw:
            return
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            self.send(self.fail(None, PARSE_ERROR, f"Parse error: {exc}"))
            return
        if isinstance(msg, list):  # (batches were dropped from the 2025-06-18 revision but are part of 2025-03-26)
            if not msg:
                self.send(self.fail(None, INVALID_REQUEST, "Invalid Request: empty batch"))
            else:
                for m in msg:
                    self.register(m)
                self.spawn(self.run_batch, msg)
            return
        if self.register(msg):
            self.spawn(lambda: self.send_opt(self.process(msg)))
        else:
            self.send_opt(self.process(msg))

    def register(self, msg) -> bool:
        """A well-formed tools/call is entered in `inflight` here, on the reading thread, before any worker starts: a notifications/cancelled
        that follows it immediately must find it. True when the message is such a call."""
        if not (isinstance(msg, dict) and msg.get("method") == "tools/call" and "id" in msg):
            return False
        rid, params = msg["id"], msg.get("params")
        if isinstance(rid, bool) or not isinstance(rid, (int, str)):
            return False
        meta = params.get("_meta") if isinstance(params, dict) and isinstance(params.get("_meta"), dict) else {}
        token = meta.get("progressToken")
        with self.lock:
            self.inflight[rid] = Ctx(self, rid, token if isinstance(token, (str, int)) and not isinstance(token, bool) else None)
        return True

    def send_opt(self, resp) -> None:
        if resp is not None:
            self.send(resp)

    def run_batch(self, batch: list) -> None:
        out = [r for r in (self.process(m) for m in batch) if r is not None]
        if out:
            self.send(out)

    def spawn(self, fn, *args) -> None:
        def body():
            try:
                fn(*args)
            except Exception:  # noqa: BLE001
                log("internal error in a worker thread:\n" + traceback.format_exc())
            finally:
                with self.lock:
                    self.threads.discard(threading.current_thread())
        t = threading.Thread(target=body, daemon=True, name="imageloom-mcp-call")
        with self.lock:
            self.threads.add(t)
        t.start()

    # -- dispatch
    def process(self, msg):
        """One JSON-RPC message in, the response out (None when none is due)."""
        if not isinstance(msg, dict):
            return self.fail(None, INVALID_REQUEST, "Invalid Request: expected a JSON object")
        has_id = "id" in msg
        rid = msg.get("id")
        if "method" not in msg and ("result" in msg or "error" in msg):
            return None  # a reply to something we never asked
        if msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return self.fail(rid if isinstance(rid, (int, str)) and not isinstance(rid, bool) else None, INVALID_REQUEST, 'Invalid Request: expected {"jsonrpc": "2.0", "method": "...", "id": ..., "params": {...}}')
        if has_id and (isinstance(rid, bool) or not isinstance(rid, (int, str))):
            return self.fail(None, INVALID_REQUEST, "Invalid Request: id must be a string or an integer")
        method, params = msg["method"], msg.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return self.fail(rid, INVALID_PARAMS, "Invalid params: expected an object") if has_id else None
        if not has_id:
            self.on_notification(method, params)
            return None
        try:
            result = self.on_request(method, params, rid)
        except RpcError as exc:
            return self.fail(rid, exc.code, exc.message, exc.data)
        except Cancelled:
            return None
        except Exception as exc:  # noqa: BLE001
            log(f"internal error handling {method}:\n" + traceback.format_exc())
            return self.fail(rid, INTERNAL_ERROR, f"Internal error: {type(exc).__name__}: {exc}")
        return None if result is None else {"jsonrpc": "2.0", "id": rid, "result": result}

    def on_notification(self, method: str, params: dict) -> None:
        debug("notification", method)
        if method == "notifications/cancelled":
            with self.lock:
                ctx = self.inflight.get(params.get("requestId"))
            if ctx is not None:
                ctx.cancelled.set()
                debug("cancelled", ctx.request_id, params.get("reason"))
        # anything else (roots/list_changed, unknown ones) is ignored, as the spec asks

    def on_request(self, method: str, params: dict, rid):
        debug("request", method, rid)
        if method == "initialize":
            return self.initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            if params.get("cursor"):
                raise RpcError(INVALID_PARAMS, "Invalid cursor: this server returns all its tools in one page")
            return {"tools": [t.describe() for t in TOOLS.values()]}
        if method == "tools/call":
            return self.call_tool(params, rid)
        raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

    def initialize(self, params: dict) -> dict:
        if self.init_seen:
            raise RpcError(INVALID_REQUEST, "initialize was already called on this connection")
        wanted = params.get("protocolVersion")
        if not isinstance(wanted, str) or not wanted:
            raise RpcError(INVALID_PARAMS, "initialize needs params.protocolVersion (a string such as '2025-06-18')")
        self.init_seen = True
        self.version = wanted if wanted in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        self.client = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {"name": "unknown"}
        info = {"name": SERVER_NAME, "version": VERSION}
        if self.version >= "2025-06-18":
            info["title"] = "ImageLoom"
        log(f"{self.client.get('name', 'client')} {self.client.get('version', '')} connected, protocol {self.version}".replace("  ", " "))
        return {"protocolVersion": self.version, "capabilities": {"tools": {"listChanged": False}}, "serverInfo": info, "instructions": INSTRUCTIONS}

    def parse_call(self, params: dict) -> tuple:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise RpcError(INVALID_PARAMS, "tools/call needs params.name (the tool's name)")
        tool_ = TOOLS.get(name)
        if tool_ is None:
            raise RpcError(INVALID_PARAMS, f"Unknown tool: {name}", {"available": sorted(TOOLS)})
        args = params.get("arguments")
        args = {} if args is None else args
        if not isinstance(args, dict):
            raise RpcError(INVALID_PARAMS, "params.arguments must be an object")
        return name, tool_, args

    def call_tool(self, params: dict, rid):
        try:
            name, tool_, args = self.parse_call(params)
        except RpcError:
            with self.lock:
                self.inflight.pop(rid, None)
            raise
        with self.lock:
            ctx = self.inflight.get(rid)
        if ctx is None:  # (not entered by register(): a call inside a malformed batch, say)
            ctx = Ctx(self, rid, None)
            with self.lock:
                self.inflight[rid] = ctx
        try:
            ctx.check()
            while not self.slots.acquire(timeout=0.25):
                ctx.check()
            try:
                res = tool_.fn(validate_args(name, tool_.input_schema, args), ctx)
            except ToolError as exc:
                res = Result(str(exc), is_error=True)
            except (Cancelled, RpcError):
                raise
            except Exception as exc:  # noqa: BLE001 - a bug in a tool must not kill the server or the conversation
                log(f"tool {name} crashed:\n" + traceback.format_exc())
                res = Result(f"internal error in {name}: {type(exc).__name__}: {exc}. This is a bug in the ImageLoom MCP server (details on its stderr); retrying may not help.", is_error=True)
            finally:
                self.slots.release()
        finally:
            with self.lock:
                self.inflight.pop(rid, None)
        if ctx.cancelled.is_set():
            return None  # the client cancelled: it no longer wants an answer
        return res.to_mcp(self.version)

    def shutdown(self, grace: float = 1.5) -> None:
        """stdin closed: waits stop at once, quick calls get a moment to answer, then we are gone."""
        self.stopping.set()
        end = time.monotonic() + grace
        with self.lock:
            threads = list(self.threads)
        for t in threads:
            t.join(max(end - time.monotonic(), 0))


def stdio_streams():
    """(reader, writer) for the protocol. fd 1 is re-pointed at stderr, so a stray print() or a child process can never corrupt the stream."""
    reader = sys.stdin.buffer  # (AttributeError when stdin is closed)
    out_fd = os.dup(1)
    try:
        os.dup2(2, 1)
    except OSError:  # (no usable stderr either: anything that prints is simply dropped)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(out_fd, os.O_BINARY)
    sys.stdout = sys.stderr
    return reader, os.fdopen(out_fd, "wb", buffering=0)


def main(argv: list | None = None, cli=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="imageloom mcp" if cli is not None else "imageloom_mcp.py",
                                description="ImageLoom as an MCP server on stdio: JSON-RPC on stdin/stdout, logs on stderr. Configure it in your MCP client; see docs/MCP.md.")
    p.add_argument("--version", action="store_true", help="print the version and exit")
    args = p.parse_args(argv)
    if args.version:
        print(f"imageloom-mcp {VERSION} (protocol {', '.join(PROTOCOL_VERSIONS)})")
        return 0
    if cli is not None:
        B.set_cli(cli)
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if sys.stdin.isatty():
        log("this is an MCP server: it speaks JSON-RPC on stdin/stdout and is meant to be started by an MCP client (see docs/MCP.md). Press Ctrl-D to quit.")
    try:
        reader, writer = stdio_streams()
    except (OSError, AttributeError) as exc:
        log(f"cannot use stdin/stdout ({exc})")
        return 1
    log(f"{SERVER_NAME} {VERSION} ready on stdio")
    server = McpServer(reader, writer)
    code = server.run()
    with contextlib.suppress(Exception):
        writer.flush()
    if any(t.is_alive() for t in list(server.threads)):  # a call stuck in a slow request must not hold the exit
        sys.stderr.flush()
        os._exit(code)
    return code


if __name__ == "__main__":
    sys.exit(main())
