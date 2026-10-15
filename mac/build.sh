#!/bin/bash
# Builds "ImageLoom.app" (Apple Silicon) and a .dmg, entirely locally.
#   ./mac/build.sh            -> dist/ImageLoom.app + dist/ImageLoom-<ver>-macOS-arm64.dmg
# Needs: Xcode command line tools (swiftc), python3 with Pillow (for the icon), network (embedded Python, wheels, sd-cli release).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VER="${VERSION:-1.0.0}"
CACHE="$ROOT/.cache"; DIST="$ROOT/dist"; STAGE="$ROOT/.build/mac"
APP="$STAGE/ImageLoom.app"; RES="$APP/Contents/Resources"
PBS_TAG="${PBS_TAG:-20261013}"; PY_VER="${PY_VER:-3.13.16}"
PY_TGZ="cpython-${PY_VER}+${PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz"
mkdir -p "$CACHE" "$DIST"; rm -rf "$STAGE"; mkdir -p "$APP/Contents/MacOS" "$RES"

echo "==> 1/7 compile the Swift shell"
swiftc -O -swift-version 5 -target arm64-apple-macos13.0 "$ROOT/mac/Sources/main.swift" \
  -o "$APP/Contents/MacOS/ImageLoom" -framework Cocoa -framework WebKit -framework UserNotifications

echo "==> 2/7 embedded Python ${PY_VER} (python-build-standalone)"
if [ ! -f "$CACHE/$PY_TGZ" ]; then
  curl -fL --retry 5 -o "$CACHE/$PY_TGZ" "https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PY_TGZ//+/%2B}"
fi
tar -xzf "$CACHE/$PY_TGZ" -C "$RES"                         # -> Resources/python
PY="$RES/python/bin/python3"
"$PY" -m pip install --quiet --no-warn-script-location --disable-pip-version-check -r "$ROOT/core/requirements.txt"
# trim what the server never uses
rm -rf "$RES/python/lib/python3.13/test" "$RES/python/lib/python3.13/idlelib" "$RES/python/lib/python3.13/tkinter" \
       "$RES/python/lib/python3.13/turtledemo" "$RES/python/lib/python3.13/ensurepip" "$RES/python/include" "$RES/python/share" \
       "$RES/python/lib/tcl9.0" "$RES/python/lib/tk9.0" "$RES/python/lib/itcl4.3.4" "$RES/python/lib/thread3.0.1" 2>/dev/null || true
find "$RES/python" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "==> 3/7 server + UI"
mkdir -p "$RES/core"
cp "$ROOT"/core/*.py "$RES/core/"; cp "$ROOT/core/imageloom" "$RES/core/imageloom"; chmod +x "$RES/core/imageloom"; cp -R "$ROOT/core/web" "$RES/core/web"
sed -i '' "s/^VERSION = .*/VERSION = \"${VER}\"/" "$RES/core/server.py"

echo "==> 4/7 engine (stable-diffusion.cpp sd-cli, official release, sha256 checked)"
ASSET_JSON="$(gh api repos/leejet/stable-diffusion.cpp/releases/latest --jq '.assets[] | select(.name|test("Darwin-macOS.*arm64.zip$")) | {n:.name,u:.browser_download_url,d:.digest}' 2>/dev/null || true)"
if [ -n "$ASSET_JSON" ]; then
  A_NAME="$(echo "$ASSET_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["n"])')"
  A_URL="$(echo "$ASSET_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["u"])')"
  A_SHA="$(echo "$ASSET_JSON" | python3 -c 'import json,sys; print((json.load(sys.stdin)["d"] or "").replace("sha256:",""))')"
  [ -f "$CACHE/$A_NAME" ] || curl -fL --retry 5 -o "$CACHE/$A_NAME" "$A_URL"
  if [ -n "$A_SHA" ] && [ "$(shasum -a 256 "$CACHE/$A_NAME" | cut -d' ' -f1)" != "$A_SHA" ]; then echo "engine checksum mismatch"; exit 1; fi
  mkdir -p "$RES/engine" && unzip -q -o "$CACHE/$A_NAME" -d "$RES/engine" && rm -f "$RES/engine/"*.txt
else  # no `gh`/network: point IMAGELOOM_ENGINE_DIR at a folder holding sd-cli and libstable-diffusion.dylib
  [ -n "${IMAGELOOM_ENGINE_DIR:-}" ] || { echo "cannot query GitHub for the engine release: install/login gh or set IMAGELOOM_ENGINE_DIR"; exit 1; }
  mkdir -p "$RES/engine" && cp "$IMAGELOOM_ENGINE_DIR/"{sd-cli,libstable-diffusion.dylib} "$RES/engine/"
fi
chmod +x "$RES/engine/sd-cli"

echo "==> 5/7 icon"
ICON="$ROOT/assets/icon_1024.png"
[ -f "$ICON" ] || python3 "$ROOT/assets/make_icon.py" "$ICON"
ISET="$STAGE/AppIcon.iconset"; mkdir -p "$ISET"
for s in 16 32 128 256 512; do
  sips -z $s $s "$ICON" --out "$ISET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s*2)) $((s*2)) "$ICON" --out "$ISET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ISET" -o "$RES/AppIcon.icns"

echo "==> 6/7 Info.plist + ad-hoc signature"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>ImageLoom</string>
  <key>CFBundleDisplayName</key><string>ImageLoom</string>
  <key>CFBundleIdentifier</key><string>io.github.elapen.imageloom</string>
  <key>CFBundleExecutable</key><string>ImageLoom</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VER}</string>
  <key>CFBundleVersion</key><string>${VER}</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
  <key>NSHumanReadableCopyright</key><string>Local AI image studio. Model weights are downloaded separately under their own licenses.</string>
</dict></plist>
PLIST
chmod -R u+rwX "$APP"
codesign --force --deep --sign - "$APP" 2>&1 | tail -2 || true
codesign --verify --deep --strict "$APP" && echo "signature ok (ad-hoc)"

echo "==> 7/7 install into dist + dmg"
rm -rf "$DIST/ImageLoom.app"; cp -R "$APP" "$DIST/ImageLoom.app"
DMGDIR="$STAGE/dmg"; mkdir -p "$DMGDIR"; cp -R "$APP" "$DMGDIR/"; ln -s /Applications "$DMGDIR/Applications"
DMG="$DIST/ImageLoom-${VER}-macOS-arm64.dmg"; rm -f "$DMG"
hdiutil create -quiet -volname "ImageLoom" -srcfolder "$DMGDIR" -fs HFS+ -format UDZO -imagekey zlib-level=9 "$DMG"
du -sh "$DIST/ImageLoom.app" "$DMG" | sed "s#$ROOT/##"
echo "done"
