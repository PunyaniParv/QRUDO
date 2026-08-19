#!/bin/bash
# Build QRUDO.app.
#
#   packaging/build_mac.sh
#
# With nothing else set, the app is ad-hoc signed: it runs on this Mac
# and on any Mac it reaches without passing through the internet's
# quarantine (git, AirDrop, a USB stick) -- and the permission prompts
# already say QRUDO, which is most of what packaging is for.
#
# Distribution to strangers needs Apple's blessing.  Once there is a
# Developer ID certificate in the keychain and a notarytool keychain
# profile (one-time: xcrun notarytool store-credentials), the same
# command does the whole ceremony:
#
#   QRUDO_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
#   QRUDO_NOTARIZE_PROFILE="qrudo" \
#   packaging/build_mac.sh
#
# and dist/QRUDO.zip is then the thing to hand to anyone.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python

if ! $PYTHON -c "import PyInstaller" 2>/dev/null; then
  $PYTHON -m pip install pyinstaller
fi

$PYTHON -m PyInstaller --noconfirm --clean packaging/qrudo.spec

APP="dist/QRUDO.app"

# Ship matplotlib's font cache pre-built.  mediapipe imports matplotlib
# during hand-tracker setup, and matplotlib builds a font cache the
# first time it is imported on a machine that has none -- which, for a
# packaged app, is the first launch after every build, on the main
# thread, for tens of seconds, before the camera ever opens.  That was
# "the camera did not switch on" / "I have to open it twice".  Building
# the cache here and copying it inside the bundle means every launch
# finds it ready.
CACHE_DIR="$($PYTHON -c 'import matplotlib as m; print(m.get_cachedir())' 2>/dev/null || true)"
if [ -n "$CACHE_DIR" ] && [ -d "$CACHE_DIR" ]; then
  # Ensure the cache actually exists (build it if this venv never has).
  $PYTHON -c "import matplotlib.font_manager" >/dev/null 2>&1 || true
  MPL_IN_APP="$APP/Contents/Resources/matplotlib"
  mkdir -p "$MPL_IN_APP"
  cp -f "$CACHE_DIR"/fontlist-*.json "$MPL_IN_APP/" 2>/dev/null || true
  echo "  bundled matplotlib font cache from $CACHE_DIR"
fi

if [ -n "${QRUDO_SIGN_IDENTITY:-}" ]; then
  codesign --force --deep --options runtime --timestamp \
    --entitlements packaging/entitlements.plist \
    -s "$QRUDO_SIGN_IDENTITY" "$APP"

  ditto -c -k --keepParent "$APP" dist/QRUDO.zip

  if [ -n "${QRUDO_NOTARIZE_PROFILE:-}" ]; then
    xcrun notarytool submit dist/QRUDO.zip \
      --keychain-profile "$QRUDO_NOTARIZE_PROFILE" --wait
    xcrun stapler staple "$APP"
    ditto -c -k --keepParent "$APP" dist/QRUDO.zip
  fi
else
  # Ad-hoc: an identity of "-" is no one, but it is consistently no
  # one, which is enough for macOS to remember permission grants.
  codesign --force --deep -s - "$APP"
fi

# The release artifacts, every build: the zip a Mac updates itself
# from, and the checksum line the updater refuses to move without.
ditto -c -k --keepParent "$APP" dist/QRUDO.zip
(cd dist && shasum -a 256 QRUDO.zip > SHA256SUMS)

echo
echo "  built: $APP"

# An installed copy is a promise to keep it current: if QRUDO lives in
# /Applications, every build refreshes it.  Deleting a running app is
# safe on macOS -- the running one keeps its files until it quits.
if [ -d "/Applications/QRUDO.app" ]; then
  rm -rf /Applications/QRUDO.app
  ditto "$APP" /Applications/QRUDO.app
  echo "  refreshed: /Applications/QRUDO.app"

  # An ad-hoc rebuild changes the code signature, and macOS pins the
  # camera grant to it: a stale entry silently DENIES the new build --
  # empty frames, the camera light off after two seconds, no prompt.
  # Resetting the entry turns that into a visible permission dialog on
  # the next launch, which a person can actually answer.  A Developer
  # ID signature will make grants survive rebuilds; until then, this.
  tccutil reset Camera com.qrudo.app >/dev/null 2>&1 \
    && echo "  camera permission reset -- next launch will ask once"

  # Accessibility dies the same quiet death on rebuild -- worse,
  # actually: the switch in System Settings still shows ON while the
  # system drops every key event, so seeking and media keys "work"
  # (the log says OK) and nothing happens.  Resetting turns that lie
  # into a clean ask; the backend refuses with instructions instead of
  # posting keys into the void.
  tccutil reset Accessibility com.qrudo.app >/dev/null 2>&1 \
    && echo "  accessibility reset -- re-enable QRUDO in System Settings" \
            "for media keys"
fi
