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

echo
echo "  built: $APP"

# An installed copy is a promise to keep it current: if QRUDO lives in
# /Applications, every build refreshes it.  Deleting a running app is
# safe on macOS -- the running one keeps its files until it quits.
if [ -d "/Applications/QRUDO.app" ]; then
  rm -rf /Applications/QRUDO.app
  ditto "$APP" /Applications/QRUDO.app
  echo "  refreshed: /Applications/QRUDO.app"
fi
