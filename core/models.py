"""Model registry, family recipes and Hugging Face link inspection for ImageLoom.

A *model* is a set of files (diffusion weights, text encoder, VAE, ...) plus a *family* recipe that says how
stable-diffusion.cpp's `sd-cli` must be called for it (flags, steps, guidance, sizes, "effort" presets).
Paste a Hugging Face link, the app works out the family, lists the weight files with sizes, adds the companion
files the family needs, downloads everything and registers the model; the model then shows up in the picker.

Only the Qwen-Image 2.1 recipe has been run end to end by the author on a base M4; the others follow the
stable-diffusion.cpp documentation and are flagged "experimental" in the UI until a run succeeded on this machine.
"""
from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
HF_HOSTS = ("huggingface.co", "www.huggingface.co")  # the only hosts that ever see the Hugging Face token
ALLOWED_HOSTS = (*HF_HOSTS, "github.com", "api.github.com")  # where a request may START; redirects (CDNs) are policed separately
QWEN_TURBO_SIGMAS = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"
MAX_API_BYTES = 20_000_000  # a Hugging Face API answer larger than this is not a model listing


# ----------------------------------------------------------------------------------------- state files
def warn(msg: str) -> None:
    print(f"imageloom: {msg}", file=sys.stderr, flush=True)


def atomic_write(path: Path, data: str | bytes, mode: int = 0o666, sync: bool = True) -> None:
    """Write `path` through a temp file in the same folder + os.replace: readers never see a half-written file, a crash keeps
    the old one, and each thread has its own temp name so concurrent writers cannot clobber each other."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), mode)
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            if sync:
                fh.flush()
                os.fsync(fh.fileno())
        for attempt in range(5):  # Windows refuses to replace a file another process (antivirus, indexer) holds open for a moment
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def quarantine(path: Path, why, copy_only: bool = False) -> None:
    """Set a damaged state file aside as `<name>.bad-<time>` (copy it when we keep using the readable part)."""
    bad = path.with_name(f"{path.name}.bad-{int(time.time())}")
    try:
        (shutil.copyfile if copy_only else os.replace)(path, bad)
        saved = f"; the original is kept as {bad.name}"
    except OSError:
        saved = ""
    warn(f"{path.name} is damaged ({why}){saved}")


def read_json(path: Path, default, kind=dict):
    """Load a JSON state file. Missing -> a copy of `default`. Damaged or of the wrong type -> the file is set aside and
    `default` is used, so a corrupt file can never stop the app from starting."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, kind):
            raise ValueError(f"expected a JSON {kind.__name__}")
        return data
    except FileNotFoundError:
        pass
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError both land here
        quarantine(path, exc)
    except OSError as exc:
        warn(f"cannot read {path}: {exc.strerror or exc}")
    return copy.deepcopy(default)


def _efforts(*levels):
    """(label, long-side px for creating, px for editing, steps or None for distilled models, hint)"""
    keys = ("label", "px", "px_edit", "steps", "hint")
    return [dict(zip(keys, lv)) for lv in levels]


# Each family: how to call sd-cli, what files it needs, sane defaults, and the effort ladder shown in the UI.
FAMILIES: dict[str, dict] = {
    "qwen-image-2.1": {
        "label": "Qwen-Image 2.1", "multiple": 32, "min_side": 512, "max_side": 2752,
        "flags": {"diffusion": "--diffusion-model", "llm": "--llm", "vae": "--vae"},
        "required": ["diffusion", "llm", "vae"], "optional": ["llm_vision"],
        "caps": {"edit": True, "alpha": True, "text": True},
        "variants": {
            "turbo": {"steps": 8, "cfg": 1.0, "sampler": "euler", "sigmas": QWEN_TURBO_SIGMAS, "fixed_steps": True},
            "base": {"steps": 24, "cfg": 5.0, "sampler": "euler", "sigmas": None, "fixed_steps": False},
        },
        "efforts": {
            "turbo": _efforts(("Fast", 768, 512, None, "draft, 4 min"), ("Balanced", 1024, 640, None, "6–7 min"),
                              ("Quality", 1536, 768, None, "slow, ~30 min"), ("Maximum", 2048, 1024, None, "very slow")),
            "base": _efforts(("Fast", 768, 512, 16, "fewer steps"), ("Balanced", 1024, 640, 24, "standard"),
                             ("Quality", 1280, 768, 36, "more steps"), ("Maximum", 1536, 1024, 50, "very slow")),
        },
        "companions": [
            {"role": "llm", "label": "Text encoder Qwen3-VL-8B Q8", "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF", "file": "Qwen3VL-8B-Instruct-Q8_0.gguf"},
            {"role": "vae", "label": "VAE Qwen-Image 2.1", "repo": "Comfy-Org/Qwen-Image-2.1", "file": "vae/qwen_image_2.1_vae_bf16.safetensors"},
            {"role": "llm_vision", "label": "Vision module (for editing)", "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF", "file": "mmproj-Qwen3VL-8B-Instruct-F16.gguf", "optional": True},
        ],
        "match": [r"qwen[-_ ]?image[-_ ]?2[._]1"], "notes": "understands Russian; text in images; instruction-based editing; transparent background",
    },
    "z-image": {
        "label": "Z-Image", "multiple": 16, "min_side": 512, "max_side": 2048,
        "flags": {"diffusion": "--diffusion-model", "llm": "--llm", "vae": "--vae"},
        "required": ["diffusion", "llm", "vae"], "optional": [],
        "caps": {"edit": False, "alpha": False, "text": True},
        "variants": {
            "turbo": {"steps": 8, "cfg": 1.0, "sampler": "euler", "sigmas": None, "fixed_steps": True},
            "base": {"steps": 30, "cfg": 4.0, "sampler": "euler", "sigmas": None, "fixed_steps": False},
        },
        "efforts": {
            "turbo": _efforts(("Fast", 768, 512, None, "draft"), ("Balanced", 1024, 640, None, "standard"),
                              ("Quality", 1280, 768, None, "more detail"), ("Maximum", 1536, 1024, None, "slow")),
            "base": _efforts(("Fast", 768, 512, 20, "fewer steps"), ("Balanced", 1024, 640, 30, "standard"),
                             ("Quality", 1280, 768, 40, "more steps"), ("Maximum", 1536, 1024, 50, "slow")),
        },
        "companions": [
            {"role": "llm", "label": "Text encoder Qwen3-4B Q4", "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"},
            {"role": "vae", "label": "VAE (Flux ae)", "repo": "Comfy-Org/z_image_turbo", "file": "split_files/vae/ae.safetensors"},
        ],
        "match": [r"z[-_ ]?image"], "notes": "fast and light; 6B parameters; understands Russian and English well",
    },
    "flux1": {
        "label": "FLUX.1", "multiple": 16, "min_side": 512, "max_side": 2048,
        "flags": {"diffusion": "--diffusion-model", "vae": "--vae", "clip_l": "--clip_l", "t5xxl": "--t5xxl"},
        "required": ["diffusion", "vae", "clip_l", "t5xxl"], "optional": [],
        "caps": {"edit": False, "alpha": False, "text": True},
        "variants": {
            "schnell": {"steps": 4, "cfg": 1.0, "sampler": "euler", "sigmas": None, "fixed_steps": True},
            "dev": {"steps": 20, "cfg": 1.0, "sampler": "euler", "sigmas": None, "fixed_steps": False, "guidance": 3.5},
        },
        "efforts": {
            "schnell": _efforts(("Fast", 768, 512, None, "4 steps"), ("Balanced", 1024, 640, None, "standard"),
                                ("Quality", 1280, 768, None, "more detail"), ("Maximum", 1536, 1024, None, "slow")),
            "dev": _efforts(("Fast", 768, 512, 12, "fewer steps"), ("Balanced", 1024, 640, 20, "standard"),
                            ("Quality", 1280, 768, 30, "more steps"), ("Maximum", 1536, 1024, 40, "slow")),
        },
        "companions": [
            {"role": "vae", "label": "VAE FLUX (ae)", "repo": "Comfy-Org/z_image_turbo", "file": "split_files/vae/ae.safetensors"},
            {"role": "clip_l", "label": "CLIP-L", "repo": "comfyanonymous/flux_text_encoders", "file": "clip_l.safetensors"},
            {"role": "t5xxl", "label": "T5-XXL (fp8, saves memory)", "repo": "comfyanonymous/flux_text_encoders", "file": "t5xxl_fp8_e4m3fn.safetensors"},
        ],
        "match": [r"flux[._-]?1", r"\bflux\b(?![._-]?2)"], "notes": "many parameters; needs Q4–Q5 quantisation on 16 GB",
    },
    "sdxl": {
        "label": "Stable Diffusion XL", "multiple": 8, "min_side": 512, "max_side": 1536, "single": True,
        "flags": {"checkpoint": "-m", "vae": "--vae"}, "required": ["checkpoint"], "optional": ["vae"],
        "caps": {"edit": False, "alpha": False, "text": False},
        "variants": {"base": {"steps": 25, "cfg": 7.0, "sampler": "euler_a", "sigmas": None, "fixed_steps": False}},
        "efforts": {"base": _efforts(("Fast", 768, 512, 15, "fewer steps"), ("Balanced", 1024, 640, 25, "standard"),
                                     ("Quality", 1280, 768, 35, "more steps"), ("Maximum", 1536, 1024, 50, "slow"))},
        "companions": [], "match": [r"sdxl", r"stable[-_ ]diffusion[-_ ]xl", r"\bxl\b"], "notes": "the classic; many ready-made styles and fine-tunes",
    },
    "sd15": {
        "label": "Stable Diffusion 1.5", "multiple": 8, "min_side": 256, "max_side": 1024, "single": True,
        "flags": {"checkpoint": "-m", "vae": "--vae"}, "required": ["checkpoint"], "optional": ["vae"],
        "caps": {"edit": False, "alpha": False, "text": False},
        "variants": {"base": {"steps": 25, "cfg": 7.0, "sampler": "euler_a", "sigmas": None, "fixed_steps": False}},
        "efforts": {"base": _efforts(("Fast", 384, 384, 15, "very fast"), ("Balanced", 512, 512, 25, "standard"),
                                     ("Quality", 640, 576, 35, "more steps"), ("Maximum", 768, 640, 50, "slow"))},
        "companions": [], "match": [r"stable[-_ ]diffusion[-_ ]v?1[-_.]?5", r"\bsd[-_]?1[._]?5\b", r"\bsd15\b"], "notes": "the lightest; 512 px",
    },
    "custom": {
        "label": "Other (manual)", "multiple": 8, "min_side": 256, "max_side": 2048,
        "flags": {"diffusion": "--diffusion-model", "checkpoint": "-m", "llm": "--llm", "vae": "--vae", "clip_l": "--clip_l", "t5xxl": "--t5xxl"},
        "required": [], "optional": [], "caps": {"edit": False, "alpha": False, "text": False},
        "variants": {"base": {"steps": 20, "cfg": 4.0, "sampler": "euler", "sigmas": None, "fixed_steps": False}},
        "efforts": {"base": _efforts(("Fast", 768, 512, 12, ""), ("Balanced", 1024, 640, 20, ""), ("Quality", 1280, 768, 30, ""), ("Maximum", 1536, 1024, 40, ""))},
        "companions": [], "match": [], "notes": "family not recognised: files and parameters are set by hand",
    },
}

# order of preference for a 16 GB machine: the biggest quant that still leaves room for the text encoder
QUANT_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q3_K", "Q2_K", "F16", "BF16"]
TEXT_ENCODER_MARKERS = ("t5xxl", "t5-xxl", "umt5", "clip_l", "clip-l", "clip_g", "qwen3", "qwen_3", "qwen2", "qwen-2", "mistral", "text_encoder", "text-encoder", "llm", "gemma")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "model"


_REPO_PART = r"[A-Za-z0-9_][A-Za-z0-9._-]{0,95}"
_REPO_RE = re.compile(rf"{_REPO_PART}/{_REPO_PART}")
BAD_LINK = "this does not look like a Hugging Face link (expected https://huggingface.co/owner/model)"


def parse_hf_url(text: str) -> tuple[str, str | None]:
    """'https://huggingface.co/owner/repo[/tree|blob|resolve/<rev>/<file>]' or 'owner/repo' -> (repo, file or None).
    Only https on huggingface.co itself is accepted: no other host, no http, no user:password@, no port."""
    text = text.strip() if isinstance(text, str) else ""
    if _REPO_RE.fullmatch(text):
        return text, None
    if not text or len(text) > 600 or re.search(r"[\s\\\x00-\x1f]", text):
        raise ValueError(BAD_LINK)
    try:
        u = urllib.parse.urlsplit(text if "://" in text else "https://" + text)
        host, port = (u.hostname or "").lower(), u.port
    except ValueError as exc:
        raise ValueError(BAD_LINK) from exc
    if u.scheme != "https" or host not in HF_HOSTS or port or u.username or u.password:
        raise ValueError(BAD_LINK)
    parts = [p for p in u.path.split("/") if p]
    if len(parts) < 2 or not _REPO_RE.fullmatch(f"{parts[0]}/{parts[1]}"):
        raise ValueError(BAD_LINK)
    file = None
    if len(parts) > 2:
        if parts[2] not in ("tree", "blob", "resolve"):
            raise ValueError(BAD_LINK)  # datasets/, spaces/, discussions ... are not model repositories
        if len(parts) > 4:
            file = urllib.parse.unquote("/".join(parts[4:]))
    return f"{parts[0]}/{parts[1]}", file


# ----------------------------------------------------------------------------------------- network
def _is_local_host(host: str) -> bool:
    host = host.lower().strip("[]").rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ipaddress.ip_address(host)  # a literal IP is never a CDN we expect; refusing it keeps redirects off the LAN
        return True
    except ValueError:
        return False


def _origin(url: str) -> tuple:
    u = urllib.parse.urlsplit(url)
    return u.scheme, (u.hostname or "").lower(), u.port or {"http": 80, "https": 443}.get(u.scheme)


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects (Hugging Face and GitHub hand downloads to CDNs) but only to https hosts, never to an IP literal or
    localhost, and never forward credentials to a different origin: urllib would otherwise re-send our Authorization header."""
    schemes = ("https",)
    allow_local = False  # (tests relax both knobs to talk to local servers)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlsplit(newurl)
        if u.scheme not in self.schemes or not u.hostname or (not self.allow_local and _is_local_host(u.hostname)):
            raise urllib.error.HTTPError(req.full_url, code, f"refused redirect to {u.scheme}://{u.hostname}", headers, fp)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _origin(newurl) != _origin(req.full_url):
            for h in ("Authorization", "Proxy-Authorization", "Cookie"):
                new.remove_header(h)
        return new


OPENER = urllib.request.build_opener(SafeRedirect)


def make_request(url: str, headers: dict | None = None, token: str | None = None) -> urllib.request.Request:
    """The token is attached only when the request itself goes to huggingface.co over https."""
    h = {"User-Agent": "ImageLoom", **(headers or {})}
    u = urllib.parse.urlsplit(url)
    if token and u.scheme == "https" and (u.hostname or "").lower() in HF_HOSTS:
        h["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=h)


def open_url(url: str, headers: dict | None = None, token: str | None = None, timeout: float = 60):
    u = urllib.parse.urlsplit(url)
    if u.scheme != "https" or (u.hostname or "").lower() not in ALLOWED_HOSTS or u.username or u.password or u.port:
        raise ValueError(f"refusing to contact {u.scheme}://{u.hostname}")
    return OPENER.open(make_request(url, headers, token), timeout=timeout)


def hf_json(path: str, token: str | None = None, timeout: int = 40):
    url = HF + path
    pages = []
    visited = set()

    while url:
        if url in visited:
            raise ValueError("Hugging Face returned a pagination loop")
        visited.add(url)
        try:
            with open_url(url, token=token, timeout=timeout) as r:
                raw = r.read(MAX_API_BYTES + 1)
                link = getattr(r, "headers", {}).get("Link", "")
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise PermissionError("the repository is private or requires accepting its terms: a Hugging Face token is needed") from exc
            if exc.code == 404:
                raise ValueError("repository not found (check the link)") from exc
            raise
        if len(raw) > MAX_API_BYTES:
            raise ValueError("Hugging Face sent an unexpectedly large answer")
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ValueError("Hugging Face did not answer with JSON") from exc

        # Only repository trees are cursor-paginated. Other API responses must keep
        # hf_json's original single-document semantics even if they contain a Link header.
        match = re.search(r'<([^>]+)>\s*;\s*rel=["\']?next["\']?', link, re.IGNORECASE) if "/tree/" in path else None
        if not match and not pages:
            return data
        if not isinstance(data, list):
            raise ValueError("Hugging Face sent an unexpected paginated answer")
        pages.extend(data)
        url = match.group(1) if match else ""

    return pages


def clean_repo_path(p) -> bool:
    """A file path from a remote listing that is safe to put in a URL and to reduce to a file name."""
    return (isinstance(p, str) and 0 < len(p) <= 300 and not p.startswith("/") and not re.search(r"[\\\x00-\x1f]", p)
            and all(seg not in ("", ".", "..") for seg in p.split("/")))


RESERVED_NAME = re.compile(r"(?i)(con|prn|aux|nul|com\d|lpt\d)(\..*)?")  # Windows device names


def safe_filename(name: str) -> str:
    """One harmless path component made from a remote file name: no directories, no leading dot/dash, no odd characters."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).replace("\\", "/").rsplit("/", 1)[-1]).lstrip(".-")
    stem, ext = os.path.splitext(base)
    base = (stem[:150] + ext[:20]) or "file"
    return "_" + base if RESERVED_NAME.fullmatch(base) else base


def list_files(repo: str, token: str | None = None) -> list[dict]:
    tree = hf_json(f"/api/models/{repo}/tree/main?recursive=true", token)
    out = []
    for f in tree if isinstance(tree, list) else []:
        if not isinstance(f, dict) or f.get("type") != "file" or not clean_repo_path(f.get("path")):
            continue
        size = (f.get("lfs") or {}).get("size", f.get("size", 0))
        out.append({"path": f["path"], "size": size if isinstance(size, int) and size >= 0 else 0})
    return out


def quant_of(name: str) -> str:
    m = re.search(r"(?:^|[-_.])((?:AD-)?(?:IQ\d(?:_[A-Z]+)?|Q\d(?:_K(?:_[SML])?|_\d)?|F16|BF16|FP8|FP16))(?=[-_.]|$)", name, re.IGNORECASE)
    return m.group(1).upper().replace("AD-", "") if m else ""


def classify(path: str) -> str | None:
    """What role a weight file most likely plays; None for files we do not need."""
    p = path.lower()
    base = p.rsplit("/", 1)[-1]
    if not base.endswith((".gguf", ".safetensors", ".sft")):
        return None
    if re.search(r"-\d{5}-of-\d{5}", base):
        return None  # sharded checkpoints are not supported by sd-cli
    if "mmproj" in base:
        return "llm_vision"
    if base in ("ae.safetensors", "ae.sft", "flux2_ae.safetensors") or re.search(r"(^|[-_/])vae([-_./]|$)", p):
        return "vae"
    if any(k in p for k in TEXT_ENCODER_MARKERS):
        return "text_encoder"
    if any(k in base for k in ("lora", "controlnet", "ip-adapter", "embedding")):
        return None
    return "diffusion"


def detect_family(repo: str, tags: list[str], files: list[dict]) -> tuple[str, str]:
    name = repo.lower() + " " + " ".join(tags).lower()
    blob = name + " " + " ".join(f["path"].lower() for f in files[:80])
    for key in ("qwen-image-2.1", "z-image", "flux1", "sdxl", "sd15"):
        if any(re.search(pat, blob) for pat in FAMILIES[key]["match"]):
            fam = key
            break
    else:
        fam = "custom"
    variant = next(iter(FAMILIES[fam]["variants"]))
    low = blob
    if fam in ("qwen-image-2.1", "z-image"):
        variant = "turbo" if "turbo" in low else "base"
    elif fam == "flux1":
        variant = "schnell" if "schnell" in low else "dev"
    return fam, variant


def pick_default(cands: list[dict], budget: float = 9.5e9) -> dict | None:
    """Best quant that fits the budget (a 16 GB Mac also has to hold the encoder), else the smallest."""
    if not cands:
        return None
    fit = [c for c in cands if c["size"] <= budget]
    clean = [c for c in fit if "0.9vae" not in c["path"].lower()]
    fit = clean or fit
    for q in QUANT_ORDER:
        for c in sorted(fit, key=lambda c: -c["size"]):
            if c["quant"] == q:
                return c
    return max(fit, key=lambda c: c["size"]) if fit else min(cands, key=lambda c: c["size"])


def inspect_repo(url: str, token: str | None = None) -> dict:
    """Everything the UI needs to offer a download: detected family, weight candidates with sizes, companion files, warnings."""
    repo, picked_file = parse_hf_url(url)
    info = hf_json(f"/api/models/{repo}", token)
    if not isinstance(info, dict):
        raise ValueError("Hugging Face sent an unexpected answer")
    files = list_files(repo, token)
    tags = [t for t in info.get("tags") or [] if isinstance(t, str)]
    fam_key, variant = detect_family(repo, tags, files)
    fam = FAMILIES[fam_key]
    warnings = []
    if info.get("gated"):
        warnings.append("the repository is gated: without a Hugging Face token the download may not start")
    if "diffusers" in tags and not any(f["path"].endswith(".gguf") for f in files):
        warnings.append("this is a Diffusers-format repository (sharded folders): sd-cli cannot read those; look for a GGUF or a single .safetensors file")
    cands = []
    diffusers_layout = any(f["path"] == "model_index.json" for f in files)
    for f in files:
        role = classify(f["path"])
        if diffusers_layout and "/" in f["path"] and not f["path"].startswith("diffusion_models/"):
            continue  # component folders of a Diffusers repo (unet/, vae/, ...) are not loadable on their own
        if role == "diffusion":
            cands.append({"path": f["path"], "size": f["size"], "quant": quant_of(f["path"]), "ext": f["path"].rsplit(".", 1)[-1].lower()})
    if picked_file:
        cands = [c for c in cands if c["path"] == picked_file] or cands
    cands.sort(key=lambda c: c["size"])
    default = pick_default(cands)
    if not cands:
        warnings.append("no usable weight files found in the repository (.gguf or a single .safetensors)")
    byname = {f["path"]: f["size"] for f in files}
    companions = []
    for c in fam["companions"]:
        size = 0
        try:
            size = next((f["size"] for f in list_files(c["repo"], token) if f["path"] == c["file"]), 0) if c["repo"] != repo else byname.get(c["file"], 0)
        except Exception:  # noqa: BLE001
            size = 0
        companions.append({**c, "size": size, "url": f"{HF}/{c['repo']}/resolve/main/{urllib.parse.quote(c['file'])}"})
    return {
        "repo": repo, "name": repo.split("/")[-1], "family": fam_key, "family_label": fam["label"], "variant": variant,
        "license": next((t.split(":", 1)[1][:80] for t in tags if t.startswith("license:")), ""), "likes": info.get("likes"), "downloads": info.get("downloads"),
        "gated": bool(info.get("gated")), "candidates": cands, "default": default["path"] if default else None, "companions": companions,
        "warnings": warnings, "supported": fam_key != "custom", "notes": fam["notes"], "experimental": fam_key != "qwen-image-2.1",
        "families": {k: v["label"] for k, v in FAMILIES.items()},
    }


def file_url(repo: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/main/{urllib.parse.quote(path)}"


def sd_args(model: dict, mode: str, steps_override: int | None = None) -> list[str]:
    """Model-specific part of the sd-cli command line (weights and sampling). The job adds prompt, size, seed and output."""
    fam = FAMILIES[model["family"]]
    params = {**fam["variants"].get(model.get("variant"), next(iter(fam["variants"].values()))), **(model.get("params") or {})}
    files = model["files"]
    args: list[str] = []
    for role, flag in fam["flags"].items():
        if files.get(role):
            args += [flag, files[role]]
    if mode == "edit" and files.get("llm_vision"):
        args += ["--llm_vision", files["llm_vision"]]
    steps = params["steps"] if params.get("fixed_steps") or not steps_override else steps_override
    args += ["--steps", str(steps), "--cfg-scale", str(params["cfg"]), "--sampling-method", params["sampler"]]
    if params.get("sigmas"):
        args += ["--sigmas", params["sigmas"]]
    if params.get("guidance") is not None:
        args += ["--guidance", str(params["guidance"])]
    return args


def effort_levels(model: dict) -> list[dict]:
    fam = FAMILIES[model["family"]]
    return fam["efforts"].get(model.get("variant")) or next(iter(fam["efforts"].values()))


def public_model(model: dict, ready: bool) -> dict:
    fam = FAMILIES[model["family"]]
    return {
        "id": model["id"], "name": model["name"], "family": model["family"], "family_label": fam["label"], "variant": model.get("variant"),
        "builtin": bool(model.get("builtin")), "ready": ready, "status": model.get("status", "ready"), "size": model.get("size", 0),
        "tags": model.get("tags", []), "license": model.get("license", ""), "source": model.get("source", ""), "notes": model.get("notes") or fam["notes"],
        "experimental": bool(model.get("experimental")), "verified": bool(model.get("verified")), "caps": {**fam["caps"], **(model.get("caps") or {})},
        "multiple": fam["multiple"], "min_side": fam["min_side"], "max_side": fam["max_side"], "efforts": effort_levels(model),
        "fixed_steps": bool((fam["variants"].get(model.get("variant")) or {}).get("fixed_steps")), "added": model.get("added"),
    }


class Registry:
    """models.json in the data folder holds the models added by the user; the built-in default is supplied by the server.

    File paths are written relative to the data folder (it can be moved, backed up or change drive letter) and are absolute in
    memory. Absolute paths inside the data folder are migrated on load; absolute paths elsewhere keep working as they are.
    One re-entrant lock guards every change, and the file is replaced atomically, so threads cannot corrupt it."""

    def __init__(self, path: Path, builtin: dict, base: Path | None = None):
        self.path, self.builtin = path, builtin
        self.base = Path(os.path.abspath(base or path.parent))
        self.lock = threading.RLock()
        self.data = {"models": [], "active": builtin["id"]}
        raw = read_json(path, {})
        rewrite, bad, seen = False, 0, {builtin["id"]}
        listed = raw.get("models", [])
        if not isinstance(listed, list):
            listed, bad = [], 1
        for entry in listed:
            m, migrated = self._load(entry)
            if m is None or m["id"] in seen:
                bad += 1
                continue
            seen.add(m["id"])
            self.data["models"].append(m)
            rewrite |= migrated
        if isinstance(raw.get("active"), str) and raw["active"] in seen:
            self.data["active"] = raw["active"]
        if bad:
            quarantine(path, f"{bad} unusable model entries skipped", copy_only=True)
        if rewrite or bad:
            self.save()

    def relative(self, p: str) -> str | None:
        """`p` as a posix path relative to the data folder, or None when it is not an absolute path inside it."""
        if not os.path.isabs(p):
            return None
        for a, b in ((Path(os.path.normpath(p)), self.base), (Path(os.path.realpath(p)), Path(os.path.realpath(self.base)))):
            try:
                rel = a.relative_to(b)
            except ValueError:
                continue
            if rel.parts:
                return rel.as_posix()
        return None

    def resolve(self, p: str) -> tuple[str, bool]:
        """Stored path -> (absolute path, True when it is an absolute path inside the data folder that must be rewritten as relative)."""
        if not os.path.isabs(p):
            return os.path.normpath(self.base / p), False
        if self.relative(p) is not None:
            return p, True
        if not os.path.exists(p):  # the data folder was moved: the same file may sit under <data>/models/...
            parts = Path(p).parts
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] == "models" and i < len(parts) - 1 and (self.base.joinpath(*parts[i:])).exists():
                    return str(self.base.joinpath(*parts[i:])), True
        return p, False

    def _load(self, m) -> tuple[dict | None, bool]:
        if not (isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"] and m.get("family") in FAMILIES and isinstance(m.get("files", {}), dict)):
            return None, False
        m, files, migrated = dict(m), {}, False
        m.setdefault("name", m["id"])
        m.pop("builtin", None)
        for role, p in (m.get("files") or {}).items():
            if isinstance(role, str) and isinstance(p, str) and p:
                files[role], inside = self.resolve(p)
                migrated |= inside
        m["files"] = files
        return m, migrated

    def save(self) -> None:
        with self.lock:
            out = copy.deepcopy(self.data)
            for m in out["models"]:
                m["files"] = {role: self.relative(p) or p for role, p in (m.get("files") or {}).items()}
            atomic_write(self.path, json.dumps(out, ensure_ascii=False, indent=1))

    def all(self) -> list[dict]:
        with self.lock:
            return [self.builtin, *self.data["models"]]

    def get(self, mid: str | None) -> dict | None:
        mid = mid or self.data.get("active")
        return next((m for m in self.all() if m["id"] == mid), None)

    def add(self, model: dict) -> dict:
        with self.lock:
            self.data["models"] = [m for m in self.data["models"] if m["id"] != model["id"]] + [model]
            self.save()
        return model

    def update(self, mid: str, **fields) -> None:
        with self.lock:
            for m in self.data["models"]:
                if m["id"] == mid:
                    m.update(fields)
            self.save()

    def remove(self, mid: str) -> dict | None:
        with self.lock:
            found = next((m for m in self.data["models"] if m["id"] == mid), None)
            self.data["models"] = [m for m in self.data["models"] if m["id"] != mid]
            if self.data.get("active") == mid:
                self.data["active"] = self.builtin["id"]
            self.save()
        return found

    def set_active(self, mid: str) -> None:
        with self.lock:
            self.data["active"] = mid
            self.save()
