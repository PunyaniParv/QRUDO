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

# --- the modern icon --------------------------------------------------
# Every neighbour (Chrome, VS Code, Notion) ships its icon as a
# compiled asset catalog -- Assets.car plus CFBundleIconName -- and
# macOS Tahoe renders THOSE consistently on every surface.  A legacy
# icns is drawn as-is in the app switcher but inset inside the system
# boundary in Finder, so the same art looked wrong somewhere no matter
# its geometry.  The catalog is compiled from the same icns, so there
# is still exactly one source of art.
# The REAL Tahoe icon: packaging/AppIcon.icon is a layered Icon
# Composer document (dark fill + the gold mark), and actool compiles
# it into Assets.car.  This is the format Chrome, VS Code and Notion
# ship, and the one macOS composes ITSELF and hands out pre-composed
# to every surface and every API -- Finder, Dock, cmd-tab, and the
# third-party bars that ask for a running app's icon.  A week of
# per-surface size fights came down to not being in this format.
# The legacy icns stays for macOS versions that predate it.
if xcrun actool --version >/dev/null 2>&1; then
  ICONTMP=$(mktemp -d)
  if xcrun actool packaging/AppIcon.icon \
      --compile "$APP/Contents/Resources" \
      --platform macosx --minimum-deployment-target 15.0 \
      --app-icon AppIcon \
      --output-partial-info-plist "$ICONTMP/partial.plist" \
      >/dev/null 2>&1 \
      && [ -f "$APP/Contents/Resources/Assets.car" ]; then
    echo "  layered icon compiled -- composed by macOS on every surface"
  else
    echo "  ! actool could not compile the layered icon; icns only"
  fi
  rm -rf "$ICONTMP"
fi

# --- a signing identity that stays put --------------------------------
# macOS pins every permission grant -- camera, Accessibility -- to the
# app's code signature.  Ad-hoc signing mints a NEW signature on every
# build, so every rebuild orphaned the grants: the camera prompt again,
# the Accessibility toggle again, forever.  That was the treadmill.
#
# The cure is any STABLE identity.  Absent a real Developer ID, a
# self-signed "QRUDO Dev" certificate is created once in the login
# keychain and used for every build after: to macOS each rebuild is
# then the same app, and a grant given once survives every update.
# (The first build signed this way may pop one keychain question --
# "codesign wants to sign" -- answer Always Allow, once.)
if [ -z "${QRUDO_SIGN_IDENTITY:-}" ]; then
  # A real Developer ID already in the keychain is the best stable
  # identity there is -- prefer it.  Local builds sign with it plainly;
  # the full hardened-runtime + notarization ceremony still belongs to
  # an explicit QRUDO_SIGN_IDENTITY release run.
  QRUDO_DEV_IDENTITY=$(security find-identity -v -p codesigning 2>/dev/null \
    | grep -o '"Developer ID Application: [^"]*"' | head -1 | tr -d '"' \
    || true)

  if [ -z "${QRUDO_DEV_IDENTITY:-}" ] \
      && ! security find-identity -p codesigning 2>/dev/null \
        | grep -q "QRUDO Dev"; then
    echo "  creating the QRUDO Dev signing certificate (one time)"
    CERTTMP=$(mktemp -d)
    openssl req -x509 -newkey rsa:2048 -keyout "$CERTTMP/key.pem" \
      -out "$CERTTMP/cert.pem" -days 3650 -nodes -subj "/CN=QRUDO Dev" \
      -addext "keyUsage=digitalSignature" \
      -addext "extendedKeyUsage=codeSigning" >/dev/null 2>&1
    openssl pkcs12 -export -out "$CERTTMP/qrudo.p12" \
      -inkey "$CERTTMP/key.pem" -in "$CERTTMP/cert.pem" \
      -passout pass:qrudo-dev >/dev/null 2>&1
    security import "$CERTTMP/qrudo.p12" \
      -k ~/Library/Keychains/login.keychain-db -P qrudo-dev -A \
      >/dev/null 2>&1 || true
    rm -rf "$CERTTMP"
  fi

  if [ -z "${QRUDO_DEV_IDENTITY:-}" ] \
      && security find-identity -p codesigning 2>/dev/null \
        | grep -q "QRUDO Dev"; then
    QRUDO_DEV_IDENTITY="QRUDO Dev"
  fi
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
elif [ -n "${QRUDO_DEV_IDENTITY:-}" ] \
    && codesign --force --deep -s "$QRUDO_DEV_IDENTITY" "$APP" 2>/dev/null; then
  STABLE_IDENTITY=1
  echo "  signed: $QRUDO_DEV_IDENTITY -- grants survive rebuilds"
else
  # Ad-hoc: the last resort, and the treadmill -- every build is a
  # stranger to macOS and the grants below have to be reset.
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
  # -dvv, not -dv: only the double-verbose form prints the Authority
  # lines this check reads, and without them every rebuild looked like
  # an identity change and reset the grants -- the exact treadmill the
  # stable identity exists to end.
  PREV_STABLE=$(codesign -dvv /Applications/QRUDO.app 2>&1 \
    | grep -c "Authority=${QRUDO_DEV_IDENTITY:-QRUDO Dev}" || true)

  rm -rf /Applications/QRUDO.app
  ditto "$APP" /Applications/QRUDO.app
  echo "  refreshed: /Applications/QRUDO.app"

  # macOS pins grants to the signature.  When this build carries the
  # same stable identity as the copy it replaced, the grants simply
  # carry over -- nothing to reset, nothing to re-answer, which is the
  # whole point of the QRUDO Dev certificate.  Only an identity CHANGE
  # (first stable build, or an ad-hoc fallback) leaves stale entries
  # that would silently deny -- empty camera frames, key events posted
  # into the void while Settings shows everything ON -- and only then
  # are they reset, so the breakage becomes a visible question a
  # person can answer once.
  if [ -n "${STABLE_IDENTITY:-}" ] && [ "${PREV_STABLE:-0}" -gt 0 ]; then
    echo "  same identity as the installed copy -- grants carry over"
  else
    tccutil reset Camera com.qrudo.app >/dev/null 2>&1 \
      && echo "  camera permission reset -- next launch will ask once"
    tccutil reset Accessibility com.qrudo.app >/dev/null 2>&1 \
      && echo "  accessibility reset -- re-enable QRUDO in System" \
              "Settings for media keys"
  fi
fi
