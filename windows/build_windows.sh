#!/bin/bash
# Cross-assembles the Windows build of ImageLoom ON A MAC: no Windows machine, no CI, nothing is executed on Windows.
#   ./windows/build_windows.sh   -> dist/ImageLoom-<ver>-Windows-x64-Setup.exe + dist/ImageLoom-<ver>-Windows-x64-portable.zip
# Needs: curl tar unzip zip shasum; Homebrew (installs makensis when missing); gh (optional, else the plain GitHub API);
#        .venv/bin/python with Pillow (for app.ico); pip (from .venv, python3 or `uvx pip`); network.
# Downloads are cached in .cache/, the staging tree is .build/windows (rebuilt on every run), artifacts go to dist/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WIN="$ROOT/windows"
VER="${VERSION:-1.0.0}"
CACHE="$ROOT/.cache"; DIST="$ROOT/dist"; STAGE="$ROOT/.build/windows"
PBS_TAG="${PBS_TAG:-20261013}"; PY_VER="${PY_VER:-3.13.16}"
PY_TGZ="cpython-${PY_VER}+${PBS_TAG}-x86_64-pc-windows-msvc-install_only.tar.gz"
PY_MM="${PY_VER%.*}"; PY_TAG="cp${PY_MM/./}"          # 3.13 -> cp313
WHEELS="$CACHE/win-wheels"
P="$STAGE/python"; SP="$P/Lib/site-packages"
BASE="ImageLoom-${VER}-Windows-x64"
SETUP="$DIST/${BASE}-Setup.exe"; ZIP="$DIST/${BASE}-portable.zip"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$VER" =~ ^[0-9]+(\.[0-9]+){0,3}([-+.][0-9A-Za-z.+-]+)?$ ]] || die "VERSION '$VER' must look like 1.2.3 (optionally with a suffix)"
NUM="${VER%%[^0-9.]*}"; IFS=. read -r v1 v2 v3 v4 <<<"$NUM"; VER_NUM="${v1:-1}.${v2:-0}.${v3:-0}.${v4:-0}"
for t in curl tar unzip zip shasum; do command -v "$t" >/dev/null || die "missing tool: $t"; done

# Python with Pillow, used for the icon and for parsing JSON / unpacking downloaded archives (always `-I`: never import from cwd)
PYV="$ROOT/.venv/bin/python"
"$PYV" -I -c "import PIL" 2>/dev/null || die "$PYV needs Pillow (pip install pillow, or recreate the venv)"

pip_cmd() {   # any pip will do: it only downloads for another platform, nothing is installed
  if "$PYV" -m pip --version >/dev/null 2>&1; then "$PYV" -m pip "$@"
  elif python3 -m pip --version >/dev/null 2>&1; then python3 -m pip "$@"
  elif command -v uvx >/dev/null 2>&1; then uvx --quiet pip "$@"
  else die "no pip found (.venv, python3 -m pip or uvx)"; fi
}
gh_api() {    # gh_api <path>: authenticated via gh when possible, anonymous otherwise
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then gh api "$1"
  else curl -fsSL -H "Accept: application/vnd.github+json" "https://api.github.com/$1"; fi
}
fetch() {     # fetch <url> <file> [sha256]: cached download with checksum verification
  local url="$1" out="$2" sha="${3:-}"
  if [ ! -f "$out" ]; then curl -fL --retry 5 -o "$out.part" "$url" || die "download failed: $url"; mv "$out.part" "$out"; fi
  if [ -n "$sha" ]; then
    [ "$(shasum -a 256 "$out" | cut -d' ' -f1)" = "$sha" ] || { rm -f "$out"; die "sha256 mismatch for $(basename "$out") (deleted from cache, run again)"; }
    echo "    sha256 ok: $(basename "$out")"
  else echo "    (no checksum published for $(basename "$out"))"; fi
}
need() { local f; for f in "$@"; do [ -e "$f" ] || die "expected file missing in the staged tree: ${f#"$STAGE"/}"; done; }

mkdir -p "$CACHE" "$DIST" "$WHEELS"
[ -n "$STAGE" ] && rm -rf "$STAGE"; mkdir -p "$STAGE"
if ! command -v makensis >/dev/null 2>&1; then
  echo "==> installing NSIS (makensis) with Homebrew"
  command -v brew >/dev/null 2>&1 || die "makensis is missing and Homebrew is not installed"
  HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 brew install makensis || { brew search nsis; die "brew install makensis failed"; }
fi

echo "==> 1/8 embedded Windows Python ${PY_VER} (python-build-standalone ${PBS_TAG}, x86_64-pc-windows-msvc)"
PY_SHA="$(gh_api "repos/astral-sh/python-build-standalone/releases/tags/${PBS_TAG}" 2>/dev/null | PY_TGZ="$PY_TGZ" "$PYV" -I -c '
import json, os, sys
a = next((x for x in json.load(sys.stdin)["assets"] if x["name"] == os.environ["PY_TGZ"]), None)
print(((a or {}).get("digest") or "").removeprefix("sha256:"))' || true)"
fetch "https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PY_TGZ//+/%2B}" "$CACHE/$PY_TGZ" "$PY_SHA"
tar -xzf "$CACHE/$PY_TGZ" -C "$STAGE"                      # -> .build/windows/python
need "$P/python.exe" "$P/pythonw.exe" "$P/python${PY_MM/./}.dll"
# prune what the server never uses: tests, IDLE, Tk, pip, headers, import libs, debug symbols
rm -rf "$P/include" "$P/libs" "$P/tcl" "$P/Scripts" "$P/Lib/test" "$P/Lib/idlelib" "$P/Lib/tkinter" "$P/Lib/turtledemo" \
       "$P/Lib/ensurepip" "$P/Lib/venv" "$P/Lib/pydoc_data" "$SP/pip" "$SP"/pip-*.dist-info
rm -f "$P"/*.pdb "$P"/DLLs/*.pdb "$P"/DLLs/_tkinter.pyd "$P"/DLLs/tcl*.dll "$P"/DLLs/tk*.dll "$P"/DLLs/_test*.pyd "$P"/DLLs/_ctypes_test.pyd
find "$P/Lib" -path "$SP" -prune -o -type d \( -name test -o -name tests \) -prune -exec rm -rf {} + 2>/dev/null || true
find "$P" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$SP"

echo "==> 2/8 Windows wheels (pip download --platform win_amd64, nothing is run on Windows)"
# Dependencies are resolved by hand: every line of both requirement files is fetched with --no-deps (see requirements-host.txt).
RUNW="$STAGE/_wheels"; mkdir -p "$RUNW"
fetch_pure_sdist() {   # packages that publish no wheel at all (proxy_tools): unpack the sdist WITHOUT running setup.py, pure Python only
  "$PYV" -I - "$1" "$SP" "$WHEELS" <<'PY'
import hashlib, io, json, sys, tarfile, urllib.request
from pathlib import Path
req, sp, cache = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
name, _, ver = req.partition("==")
meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json" if ver else f"https://pypi.org/pypi/{name}/json", timeout=60))
sd = next((u for u in meta["urls"] if u["packagetype"] == "sdist"), None)
if not sd:
    sys.exit(f"{req}: PyPI has neither a win_amd64/cp313 wheel nor an sdist")
dest = cache / sd["filename"]
if not dest.exists():
    dest.write_bytes(urllib.request.urlopen(sd["url"], timeout=120).read())
data = dest.read_bytes()
if hashlib.sha256(data).hexdigest() != sd["digests"]["sha256"]:
    dest.unlink()
    sys.exit(f"{req}: sdist sha256 mismatch")
tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
files = [m for m in tf.getmembers() if m.isfile()]
native = [m.name for m in files if m.name.lower().endswith((".c", ".cc", ".cpp", ".cxx", ".pyx", ".rs", ".f", ".f90", ".go", ".cu"))]
if native:
    sys.exit(f"{req}: the sdist has native sources ({native[:3]}); it cannot be cross-built here")
tl = next((m for m in files if m.name.endswith(".egg-info/top_level.txt")), None)
tops = tf.extractfile(tl).read().decode().split() if tl else sorted({m.name.split("/")[1] for m in files if m.name.count("/") == 2 and m.name.endswith("/__init__.py")})
if not tops or not all(t.isidentifier() for t in tops):
    sys.exit(f"{req}: cannot tell which packages the sdist provides")
n = 0
for m in files:
    rel = m.name.split("/", 1)[1] if "/" in m.name else ""
    if any(rel == t + ".py" or rel.startswith(t + "/") for t in tops) and ".." not in rel.split("/"):
        out = sp / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(tf.extractfile(m).read())
        n += 1
print(f"    {req}: no wheel on PyPI, unpacked {n} pure-Python files from {sd['filename']} into site-packages")
PY
}
while IFS= read -r req; do
  req="${req%%#*}"; req="$(echo "$req" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"; [ -n "$req" ] || continue
  if ! pip_cmd download --quiet --disable-pip-version-check --platform win_amd64 --python-version "$PY_MM" --implementation cp --abi "$PY_TAG" \
        --only-binary=:all: --no-deps --find-links "$WHEELS" -d "$RUNW" "$req" 2>"$STAGE/pip.err"; then
    if grep -q "No matching distribution" "$STAGE/pip.err"; then fetch_pure_sdist "$req"; else cat "$STAGE/pip.err" >&2; die "pip download failed for $req"; fi
  fi
done < <({ cat "$ROOT/core/requirements.txt"; echo; cat "$WIN/requirements-host.txt"; })
for w in "$RUNW"/*.whl; do
  case "$(basename "$w")" in   # only wheels that run on Windows x64 / CPython 3.13 are acceptable
    *-none-any.whl|*-"$PY_TAG"-"$PY_TAG"-win_amd64.whl|*-abi3-win_amd64.whl) ;;
    *) die "unexpected wheel tag: $(basename "$w")" ;;
  esac
  unzip -q -o "$w" -d "$SP"; cp -n "$w" "$WHEELS/" 2>/dev/null || true
done
for d in "$SP"/*.data; do   # pip would put .data/scripts into Scripts\ (console launchers, unused here); anything else would need real handling
  [ -d "$d" ] || continue
  [ -z "$(ls -A "$d" | grep -v '^scripts$' || true)" ] || die "wheel data dir $(basename "$d") has content other than scripts/"
  rm -rf "$d"
done
( cd "$RUNW" && ls -1 *.whl | sed 's/^/    /' )
need "$SP/PIL/__init__.py" "$SP"/PIL/_imaging."$PY_TAG"-win_amd64.pyd "$SP/send2trash/win/legacy.py" "$SP"/vtracer/vtracer."$PY_TAG"-win_amd64.pyd \
     "$SP/webview/__init__.py" "$SP/webview/lib/Microsoft.Web.WebView2.Core.dll" "$SP/webview/lib/Microsoft.Web.WebView2.WinForms.dll" \
     "$SP/webview/lib/runtimes/win-x64/native/WebView2Loader.dll" "$SP/pythonnet/runtime/Python.Runtime.dll" "$SP/clr.py" \
     "$SP/clr_loader/ffi/dlls/amd64/ClrLoader.dll" "$SP"/_cffi_backend."$PY_TAG"-win_amd64.pyd "$SP/cffi/__init__.py" "$SP/pycparser/__init__.py" \
     "$SP/proxy_tools/__init__.py" "$SP/bottle.py" "$SP/typing_extensions.py"
echo "    site-packages verified (PIL, send2trash, vtracer, webview + WebView2 DLLs, pythonnet, clr_loader, cffi, ...)"

echo "==> 3/8 engine (stable-diffusion.cpp, Vulkan build for Windows x64)"
ENG_INFO="$(gh_api "repos/leejet/stable-diffusion.cpp/releases/latest" 2>/dev/null | "$PYV" -I -c '
import json, re, sys
rel = json.load(sys.stdin)
a = next((x for x in rel["assets"] if re.search(r"bin-win-vulkan-x64\.zip$", x["name"])), None)
if a: print("|".join([a["name"], a["browser_download_url"], (a.get("digest") or "").removeprefix("sha256:"), rel["tag_name"]]))' || true)"
if [ -n "$ENG_INFO" ]; then
  IFS='|' read -r E_NAME E_URL E_SHA E_TAG <<<"$ENG_INFO"
  echo "    release $E_TAG: $E_NAME"
  fetch "$E_URL" "$CACHE/$E_NAME" "$E_SHA"; ENG_ZIP="$CACHE/$E_NAME"
else
  ENG_ZIP="$(ls -t "$CACHE"/sd-*-bin-win-vulkan-x64.zip 2>/dev/null | head -1 || true)"
  [ -n "$ENG_ZIP" ] || die "cannot reach the stable-diffusion.cpp release and no cached bin-win-vulkan-x64.zip exists"
  echo "    WARNING: GitHub unreachable, using the cached $(basename "$ENG_ZIP") without re-checking its checksum"
fi
ENG="$STAGE/engine"; mkdir -p "$ENG"
unzip -q -o "$ENG_ZIP" -d "$ENG"
# keep the upstream licence texts (MIT requires them next to redistributed binaries) out of the way in licenses/
mkdir -p "$ENG/licenses"; for f in "$ENG"/*.txt "$ENG"/*.md; do if [ -e "$f" ]; then mv "$f" "$ENG/licenses/"; fi; done
need "$ENG/sd-cli.exe" "$ENG/sd-server.exe" "$ENG/stable-diffusion.dll" "$ENG/ggml-vulkan.dll"

echo "==> 4/8 app code (server + UI + Windows host)"
mkdir -p "$STAGE/app/core"
cp "$ROOT"/core/*.py "$STAGE/app/core/"             # server.py and the sibling modules it imports (models.py, imageloom_mcp.py, ...)
cp "$ROOT/core/imageloom" "$STAGE/app/core/imageloom"  # the CLI (also imported by the MCP server)
cp -R "$ROOT/core/web" "$STAGE/app/core/web"
cp "$WIN/app.py" "$STAGE/app/app.py"
sed -i.bak "s/^VERSION = .*/VERSION = \"${VER}\"/" "$STAGE/app/core/server.py" && rm -f "$STAGE/app/core/server.py.bak"
grep -q "^VERSION = \"${VER}\"$" "$STAGE/app/core/server.py" || die "could not patch VERSION in server.py"
find "$STAGE/app" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
need "$STAGE/app/core/server.py" "$STAGE/app/core/web/index.html"
# import closure: every top-level import of the staged code must be stdlib, a sibling module, or present in the staged site-packages
"$PYV" -I - "$STAGE/app/core" "$SP" <<'PY' || die "the staged server imports something that is not packaged (see above)"
import ast, sys
from pathlib import Path
core, sp = Path(sys.argv[1]), Path(sys.argv[2])
known = set(sys.stdlib_module_names) | {p.stem for p in core.glob("*.py")} | {p.name.split(".")[0] for p in sp.iterdir() if not p.name.endswith(".dist-info")}
missing = {}
for f in sorted(core.glob("*.py")):
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
        for n in names:
            if n.split(".")[0] not in known:
                missing.setdefault(n.split(".")[0], set()).add(f.name)
if missing:
    sys.exit("    unresolved imports: " + "; ".join(f"{k} (in {', '.join(sorted(v))})" for k, v in sorted(missing.items())))
print("    import closure ok:", ", ".join(sorted(p.name for p in core.glob("*.py"))))
PY

echo "==> 5/8 icon (app.ico from assets/icon_1024.png, regenerated on every run)"
ICON="$ROOT/assets/icon_1024.png"
[ -f "$ICON" ] || "$PYV" -I "$ROOT/assets/make_icon.py" "$ICON"
"$PYV" -I - "$ICON" "$STAGE/app/app.ico" <<'PY'
import sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGBA")
box = im.getchannel("A").point(lambda v: 255 if v > 128 else 0).getbbox()   # the tile itself, without its soft shadow
if box:   # the macOS artwork keeps ~10% transparent margin; Windows icons are drawn edge to edge, so crop to the artwork + 3%
    l, t, r, b = box
    side = max(r - l, b - t); side += 2 * round(side * 0.03)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im.crop(box), ((side - (r - l)) // 2, (side - (b - t)) // 2))
    im = canvas
sizes = [16, 24, 32, 48, 64, 128, 256]
im.save(dst, format="ICO", sizes=[(s, s) for s in sizes])
got = sorted(Image.open(dst).info["sizes"])
assert got == [(s, s) for s in sizes], got
print("    app.ico sizes:", ", ".join(f"{w}" for w, _ in got))
PY

echo "==> 6/8 portable zip"
ZROOT="$ROOT/.build/windows-zip"; ZDIR="$ZROOT/${BASE}"; rm -rf "$ZROOT"; mkdir -p "$ZDIR"   # outside STAGE on purpose
for d in python app engine; do ln -s "$STAGE/$d" "$ZDIR/$d"; done     # zip follows the links, so the tree is not copied twice
printf '%s\r\n' '@echo off' 'rem ImageLoom (portable). Data, models and logs live in %LOCALAPPDATA%\ImageLoom;' \
  'rem to keep everything next to this file instead, remove "rem " from the next line.' \
  'rem set IMAGELOOM_HOME=%~dp0data' 'cd /d "%~dp0"' 'start "" "%~dp0python\pythonw.exe" "%~dp0app\app.py"' > "$ZDIR/ImageLoom.bat"
rm -f "$ZIP"; ( cd "$ZROOT" && zip -q -r -X "$ZIP" "$BASE" )
unzip -Z1 "$ZIP" > "$STAGE/zip.lst"
for f in "python/python.exe" "python/pythonw.exe" "python/Lib/site-packages/webview/__init__.py" "engine/sd-cli.exe" "app/app.py" "app/core/server.py" "ImageLoom.bat"; do
  grep -qxF "$BASE/$f" "$STAGE/zip.lst" || die "portable zip lacks $f"
done

echo "==> 7/8 NSIS installer"
KB="$(du -sk "$STAGE/python" "$STAGE/app" "$STAGE/engine" | awk '{s+=$1} END{print s}')"
rm -f "$SETUP"
# makensis 3.13 (Homebrew, arm64) dies with std::bad_alloc in "Generating language tables" under the plain C locale, which is what
# non-interactive shells give it; a UTF-8 locale fixes it.
LC_ALL=en_US.UTF-8 makensis -V2 -INPUTCHARSET UTF8 -DVERSION="$VER" -DVERSION_NUM="$VER_NUM" -DSTAGE="$STAGE" -DOUTFILE="$SETUP" -DINSTALLED_KB="$KB" "$WIN/installer.nsi"
[ -s "$SETUP" ] || die "makensis produced no installer"

echo "==> 8/8 artifacts"
ls -lh "$SETUP" "$ZIP" | awk '{print "    "$5"  "$9}' | sed "s#$ROOT/##"
echo "    staged tree: $(du -sh "$STAGE/python" "$STAGE/app" "$STAGE/engine" | awk '{printf "%s %s  ", $1, $2}' | sed "s#$STAGE/##g")"
echo "done. NOT TESTED ON WINDOWS: assembled and compiled on macOS only."
