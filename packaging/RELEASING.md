# Shipping an update to customers

Customers do not have the repository; they have whatever was frozen
into their app on build day.  The app checks the release feed at
startup and shows "QRUDO x.y.z is out" on its own window when it is
behind -- so a release is what makes every customer's copy know to
update.

The ritual, in order:

1. **Bump the number** in `version.py` (say `0.2.0`) and commit.
2. **Tag it to match**: `git tag v0.2.0 && git push --tags`.
   The tag is what the update check compares against -- the `v` prefix
   is expected, and the number must be plain digits and dots.
3. **Build both platforms** from that commit:
   - Mac: `packaging/build_mac.sh` with the signing variables set
     (see its header) -> `dist/QRUDO.zip`, notarized.
   - Windows: `packaging\update_windows.bat` on the Windows machine
     -> `dist\QRUDO-Setup.exe`.
4. **Publish a GitHub release** on the tag and attach both files.
   The moment it is published, every running copy older than it
   starts saying so.

Notes:

- The feed is `updates.LATEST_URL`; releases must be publicly
  readable for customers' apps to see them.  Moving distribution to a
  proper website later means changing that one constant.
- Stage two -- the app downloading and replacing itself -- waits on
  code signing on both platforms, because an app that swaps its own
  binaries must be able to prove the new one came from us.
