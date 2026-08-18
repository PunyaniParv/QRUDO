# QRUDO — the commands that matter, Mac and Windows

Everything below is one-time setup or the single update command. Copy-paste,
in order.

---

## macOS

### First build + install (once)
```bash
git clone https://github.com/PunyaniParv/QRUDO.git
cd QRUDO
packaging/build_mac.sh
cp -R dist/QRUDO.app /Applications/
```
Open it from Applications (or Spotlight → "QRUDO"). Grant Camera when asked;
if play/pause is dead, System Settings → Privacy & Security → Accessibility →
enable QRUDO.

### Stay updated (one command, any time)
```bash
cd QRUDO
git pull && packaging/build_mac.sh
```
`build_mac.sh` automatically refreshes `/Applications/QRUDO.app`, so this one
line pulls the newest code, rebuilds, and replaces the installed app.

### Optional — auto-update daily, hands-off
```bash
crontab -l 2>/dev/null | grep -v qrudo-update > /tmp/ct; \
echo "0 13 * * * cd $HOME/QRUDO && git pull -q && packaging/build_mac.sh >/dev/null 2>&1 # qrudo-update" >> /tmp/ct; \
crontab /tmp/ct
```
Runs the update every day at 1 pm. Remove it with `crontab -e` (delete the
`qrudo-update` line).

---

## Windows

### First build + install (once)
```
git clone https://github.com/PunyaniParv/QRUDO.git
cd QRUDO
packaging\build_windows.bat
```
Result: `dist\QRUDO\QRUDO.exe` — double-click to run. If Windows shows a blue
"protected your PC" box, click **More info → Run anyway** (normal for an
unsigned app). Needs Python from python.org first, with "Add to PATH" ticked.

### Optional extra 1 — a real installer (Start Menu, uninstall)
Install Inno Setup (https://jrsoftware.org/isinfo.php), then:
```
packaging\build_windows.bat
```
Now you also get `dist\QRUDO-Setup.exe` — the file to hand to a Windows user.

### Optional extra 2 — stay updated (one command)
```
packaging\update_windows.bat
```
Pulls the newest code, rebuilds only if something changed, and keeps a Start
Menu shortcut named QRUDO pointed at the result.

### Optional — auto-update daily, hands-off
```
schtasks /create /tn "QRUDO Update" /tr "%CD%\packaging\update_windows.bat" /sc daily /st 13:00
```
Run this from inside the QRUDO folder. Windows then updates QRUDO every day at
1 pm. Check it with `schtasks /query /tn "QRUDO Update"`.

---

## The gestures (identical on both platforms)

| Gesture | Does |
|---|---|
| Fist | Play / pause |
| Point at camera | Switch which app is controlled |
| Swipe left / right | Rewind / forward ~10s |
| Two fingers up / down | Volume up / down |
| Open palm up / down | Brightness up / down |
| `Ctrl+Shift+← / →` (keys) | Step the target app by hand |

The recognition engine is the same code on Mac and Windows — only how the
camera opens differs. So a gesture that works on one works on the other.
