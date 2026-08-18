"""The application around the engine: pages, buttons, a corner of camera.

Rough on purpose, honest throughout: this is a shell over exactly the
same loop the classic window runs -- the vision thread is
integration.runner with a frame tap and a stop flag, so nothing about
recognition, reliability or logging is different here.  Two pages,
and every corner of both does something:

    Home      top-left     what this is and which version
              top-right    Settings
              centre       the gesture and what it just did, large
              bottom-left  Pause / Resume (dry-run: watch, touch nothing)
              bottom-right Quit, with the camera preview above it

    Settings  top-left     Back
              top-right    Check for updates (installs, when frozen)
              centre       the config that matters day to day
              bottom-left  Open the logs folder
              bottom-right Save

Tk owns the main thread -- macOS insists -- so the camera loop runs on
a worker and hands each frame across in a plain attribute swap, which
the GIL makes safe.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk

BACKGROUND = "#101418"
PANEL = "#1a2027"
INK = "#e8edf2"
DIM = "#8a97a3"
ACCENT = "#4cc38a"

PREVIEW_SIZE = (320, 240)


def run_app(engine, args):
    """Build the window, start the vision thread, hand Tk the reins."""

    args.no_window = True

    app = App(engine, args)

    return app.run()


def apply_settings(config, fields):
    """Typed values from the settings page onto the live config.

    The page hands everything over as strings; each lands as whatever
    type the config already holds there, and a value that will not
    convert leaves the old one standing rather than crashing a save.
    Returns the names that changed.
    """

    changed = []

    for name, raw in fields.items():
        current = getattr(config, name, None)

        if current is None:
            continue

        try:
            if isinstance(current, bool):
                value = bool(raw) if not isinstance(raw, str) \
                    else raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                value = int(str(raw).strip())
            elif isinstance(current, float):
                value = float(str(raw).strip())
            else:
                value = str(raw).strip()
        except (TypeError, ValueError):
            continue

        if value != current:
            setattr(config, name, value)
            changed.append(name)

    return changed


class App:
    #: The settings the page offers, with the words a person reads.
    SETTINGS = [
        ("volume_step", "Volume step (%)"),
        ("brightness_step", "Brightness step (%)"),
        ("seek_seconds", "Seek jump (seconds)"),
        ("cooldown_seconds", "Wait between commands (s)"),
        ("target_app", "App to control (blank = auto)"),
    ]

    def __init__(self, engine, args):
        self.engine = engine
        self.args = args
        self.latest = None          # (frame, gesture, result, hint)
        self.stop = threading.Event()
        self.worker = None
        self.preview_photo = None   # kept, or Tk garbage-collects it
        self.ready = None           # a staged update, once one is
        self.beat = 0               # frames seen; drives the hidden dot
        self.first_frame_seen = False

        from version import VERSION

        self.root = tk.Tk()
        self.root.title("QRUDO")
        self.root.geometry("880x560")
        self.root.minsize(700, 480)
        self.root.configure(bg=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.home = tk.Frame(self.root, bg=BACKGROUND)
        self.settings = tk.Frame(self.root, bg=BACKGROUND)
        self.add_gesture = tk.Frame(self.root, bg=BACKGROUND)

        for page in (self.home, self.settings, self.add_gesture):
            page.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._build_home(VERSION)
        self._build_settings()
        self._build_add_gesture()
        self.home.tkraise()

    # -- pages ---------------------------------------------------------

    def _corner(self, page, text, command, relx, rely, anchor):
        """A button that is actually visible on every platform.

        Not tk.Button: macOS draws those in native Aqua, ignoring the
        dark background entirely -- which left near-white text on a
        white face, and a person clicking buttons they could not see.
        A Label with a click binding obeys its colours everywhere.
        """

        button = tk.Label(page, text=text, bg=PANEL, fg=INK,
                          font=("Helvetica", 14), padx=16, pady=8,
                          cursor="pointinghand" if sys.platform == "darwin"
                          else "hand2")
        # Nudge off the very edge, except a centred button which wants no
        # horizontal nudge at all.
        nudge_x = 0 if relx == 0.5 else (16 if relx == 0 else -16)
        button.place(relx=relx, rely=rely, anchor=anchor,
                     x=nudge_x,
                     y=(14 if rely == 0 else -14))

        button.bind("<Button-1>", lambda _event: command())
        button.bind("<Enter>", lambda _e: button.configure(bg=ACCENT,
                                                           fg=BACKGROUND))
        button.bind("<Leave>", lambda _e: button.configure(bg=PANEL,
                                                           fg=INK))
        return button

    def _build_home(self, version):
        tk.Label(self.home, text=f"QRUDO  {version}",
                 bg=BACKGROUND, fg=DIM, font=("Helvetica", 13)).place(
            relx=0, rely=0, anchor="nw", x=18, y=16)

        self._corner(self.home, "Settings", self.show_settings,
                     relx=1, rely=0, anchor="ne")
        self.pause_button = self._corner(
            self.home,
            "Resume" if self.engine.config.dry_run else "Pause",
            self.toggle_pause, relx=0, rely=1, anchor="sw")
        self._corner(self.home, "Quit", self.quit,
                     relx=1, rely=1, anchor="se")

        # The words keep to the left three-fifths of the window, so
        # nothing ever runs underneath the preview in the right corner.
        self.gesture_label = tk.Label(self.home, text="--",
                                      bg=BACKGROUND, fg=INK,
                                      font=("Helvetica", 44, "bold"))
        self.gesture_label.place(relx=0.36, rely=0.32, anchor="center")

        self.result_label = tk.Label(self.home, text="starting camera...",
                                     bg=BACKGROUND, fg=DIM,
                                     font=("Helvetica", 18),
                                     wraplength=420, justify="center")
        self.result_label.place(relx=0.36, rely=0.48, anchor="center")

        self.hint_label = tk.Label(self.home, text="", bg=BACKGROUND,
                                   fg=DIM, font=("Helvetica", 13),
                                   wraplength=460, justify="left")
        self.hint_label.place(relx=0, rely=1, anchor="sw", x=18, y=-72)

        # The camera, in the corner: above the Quit button, small on
        # purpose -- the product is the gestures, not the video feed.
        # Click it to hide it entirely; the gestures never needed the
        # picture, only the person did, and some people would rather
        # not watch themselves.  The choice is kept in the config.
        self.preview = tk.Label(self.home, bg=PANEL, fg=DIM,
                                font=("Helvetica", 12),
                                cursor="pointinghand"
                                if sys.platform == "darwin" else "hand2")
        self.preview.place(relx=1, rely=1, anchor="se", x=-16, y=-64)
        self.preview.bind("<Button-1>", lambda _e: self.toggle_preview())
        self._dress_preview()

    def _dress_preview(self):
        if self.engine.config.show_preview:
            self.preview.configure(text="", padx=0, pady=0)
        else:
            # "Camera hidden" read as "camera off", and it was neither:
            # only the picture goes away.  So the chip says what is
            # true and proves it -- the dot beats on real frames, so a
            # person watches the camera being alive without seeing it.
            self.preview.configure(image="",
                                   text="● watching -- picture hidden "
                                        "(click to show)",
                                   padx=12, pady=8, width=0, height=0)
            self.preview_photo = None

    def toggle_preview(self):
        config = self.engine.config
        config.show_preview = not config.show_preview
        self._dress_preview()

        try:
            config.save()
        except OSError:
            pass

    def _build_settings(self):
        self._corner(self.settings, "Back", self.show_home,
                     relx=0, rely=0, anchor="nw")
        self.update_button = self._corner(self.settings,
                                          "Check for updates",
                                          self.check_updates,
                                          relx=1, rely=0, anchor="ne")
        self._corner(self.settings, "Open logs", self.open_logs,
                     relx=0, rely=1, anchor="sw")
        self._corner(self.settings, "Save", self.save_settings,
                     relx=1, rely=1, anchor="se")

        # The one that opens the three-box form.  Centred at the bottom
        # so it is plainly a different kind of thing than the config
        # rows above it.
        self._corner(self.settings, "+ Add Gesture", self.show_add_gesture,
                     relx=0.5, rely=1, anchor="s")

        form = tk.Frame(self.settings, bg=BACKGROUND)
        form.place(relx=0.5, rely=0.45, anchor="center")

        self.fields = {}

        for row, (name, words) in enumerate(self.SETTINGS):
            tk.Label(form, text=words, bg=BACKGROUND, fg=INK,
                     font=("Helvetica", 14), anchor="e", width=26).grid(
                row=row, column=0, padx=8, pady=7, sticky="e")

            entry = tk.Entry(form, bg=PANEL, fg=INK,
                             insertbackground=INK, relief="flat",
                             font=("Helvetica", 14), width=16)
            entry.insert(0, str(getattr(self.engine.config, name, "")))
            entry.grid(row=row, column=1, padx=8, pady=7, sticky="w")
            self.fields[name] = entry

        self.settings_note = tk.Label(self.settings, text="",
                                      bg=BACKGROUND, fg=ACCENT,
                                      font=("Helvetica", 13))
        self.settings_note.place(relx=0.5, rely=0.87, anchor="center")

    def _build_add_gesture(self):
        """The three-box form: what work, which gesture, which app.

        Deliberately plain and readable -- this is the screen the user
        asked to see so they can react to it.  The 'work' box is the
        catalog; the 'gesture' box will record a shape (wired next); the
        'app' box names where the action lands, with the global/locked
        choice.
        """

        from tkinter import ttk

        from control import catalog

        self._corner(self.add_gesture, "Back", self.show_settings,
                     relx=0, rely=0, anchor="nw")
        self._corner(self.add_gesture, "Save gesture", self.save_gesture,
                     relx=1, rely=1, anchor="se")

        tk.Label(self.add_gesture, text="Teach QRUDO a new gesture",
                 bg=BACKGROUND, fg=INK, font=("Helvetica", 20, "bold")).place(
            relx=0.5, rely=0.1, anchor="center")

        form = tk.Frame(self.add_gesture, bg=BACKGROUND)
        form.place(relx=0.5, rely=0.48, anchor="center")

        def row(r, label):
            tk.Label(form, text=label, bg=BACKGROUND, fg=INK,
                     font=("Helvetica", 14), anchor="e", width=22).grid(
                row=r, column=0, padx=8, pady=10, sticky="e")

        # Box 1 -- what is the work
        row(0, "1.  What should it do?")
        self.job_choice = ttk.Combobox(form, values=catalog.job_names(),
                                       state="readonly", width=24)
        self.job_choice.set("Next track")
        self.job_choice.grid(row=0, column=1, padx=8, pady=10, sticky="w")

        # Box 2 -- which gesture (recording wired in the next phase)
        row(1, "2.  Which gesture?")
        self.gesture_status = tk.Label(
            form, text="record it  (coming: hold your shape to the camera)",
            bg=PANEL, fg=DIM, font=("Helvetica", 13), padx=12, pady=8,
            cursor="pointinghand" if sys.platform == "darwin" else "hand2")
        self.gesture_status.grid(row=1, column=1, padx=8, pady=10, sticky="w")
        self.gesture_status.bind("<Button-1>",
                                 lambda _e: self.record_gesture())

        # Box 3 -- which app, and whether locked
        row(2, "3.  Which app?")
        self.app_choice = ttk.Combobox(
            form, values=["YouTube Music", "Spotify", "Front app (auto)"],
            state="readonly", width=24)
        self.app_choice.set("YouTube Music")
        self.app_choice.grid(row=2, column=1, padx=8, pady=10, sticky="w")

        # Scope -- where the gesture applies
        row(3, "Where it works:")
        self.scope_choice = ttk.Combobox(
            form, values=["Global (anywhere)",
                          "Only when that app is in front"],
            state="readonly", width=24)
        self.scope_choice.set("Global (anywhere)")
        self.scope_choice.grid(row=3, column=1, padx=8, pady=10, sticky="w")

        tk.Label(self.add_gesture,
                 text="Tip: Ctrl+Shift+←/→ switches the target app "
                      "live, any time.",
                 bg=BACKGROUND, fg=DIM, font=("Helvetica", 12)).place(
            relx=0.5, rely=0.82, anchor="center")

        self.add_note = tk.Label(self.add_gesture, text="", bg=BACKGROUND,
                                 fg=ACCENT, font=("Helvetica", 13))
        self.add_note.place(relx=0.5, rely=0.9, anchor="center")

        #: Filled once a shape is recorded.
        self._recorded_signature = None

    # -- what the corners do -------------------------------------------

    def show_add_gesture(self):
        self.add_gesture.tkraise()

    def record_gesture(self):
        """Record the shape held to the camera.  Wired in the next phase;
        for now it tells the truth about that rather than pretending."""

        self.add_note.configure(
            text="gesture recording is the next piece -- the form is here "
                 "first so you can shape it")

    def save_gesture(self):
        """Turn the three boxes into a saved custom gesture.

        The work and app resolve through the catalog; the shape comes
        from recording.  Without a recorded shape yet, this explains
        what is missing rather than saving half a gesture.
        """

        from control import catalog

        job = self.job_choice.get()
        app_label = self.app_choice.get()
        locked = app_label != "Front app (auto)"

        app_key = {"YouTube Music": "youtube_music",
                   "Spotify": "spotify"}.get(app_label, "any")

        action = catalog.resolve(job, app_key, lock_to_app=locked)

        if action is None:
            self.add_note.configure(
                text=f"no built-in shortcut for {job!r} on {app_label} yet")
            return

        if self._recorded_signature is None:
            self.add_note.configure(
                text=f"ready: {job} → {app_label}. Now record a gesture "
                     f"(step 2) to save it.")
            return

        self.add_note.configure(text="(saving wired in the next phase)")

    def show_settings(self):
        self.settings.tkraise()

    def show_home(self):
        self.home.tkraise()

    def toggle_pause(self):
        config = self.engine.config
        config.dry_run = not config.dry_run
        self.pause_button.configure(
            text="Resume" if config.dry_run else "Pause")

    def save_settings(self):
        values = {name: entry.get() for name, entry in self.fields.items()}
        changed = apply_settings(self.engine.config, values)

        try:
            self.engine.config.save()
        except OSError as exc:
            self.settings_note.configure(text=f"could not save: {exc}")
            return

        self.settings_note.configure(
            text=("saved: " + ", ".join(changed)) if changed
            else "saved (nothing changed)")

    def open_logs(self):
        from paths import resolve

        folder = resolve(self.engine.config.log_dir)
        folder.mkdir(parents=True, exist_ok=True)
        opener = "open" if sys.platform == "darwin" else "explorer"
        subprocess.run([opener, str(folder)], check=False)

    def check_updates(self):
        self.update_button.configure(text="checking...")

        def ask():
            import updates

            latest = updates.check()

            if not latest:
                verdict = f"current ({updates.VERSION})"
            elif getattr(sys, "frozen", False):
                import selfupdate

                ready = selfupdate.prepare()

                if ready:
                    self.ready = ready
                    verdict = f"install {latest} (restarts)"
                    self.update_button.configure(command=self.install_update)
                else:
                    verdict = f"{latest} is out -- see releases page"
            else:
                verdict = f"{latest} is out -- git pull"

            self.root.after(0, lambda: self.update_button.configure(text=verdict))

        threading.Thread(target=ask, daemon=True).start()

    def install_update(self):
        from pathlib import Path

        import selfupdate

        staged, version = self.ready
        self.quit()

        if selfupdate.apply(staged, version):
            selfupdate.relaunch(Path(sys.executable).resolve().parents[2])

    def quit(self):
        self.stop.set()

        if self.worker is not None:
            self.worker.join(timeout=3.0)

        self.root.destroy()

    # -- the vision thread and the pulse -------------------------------

    def take_frame(self, frame, gesture, result, hint):
        self.latest = (frame, gesture, result, hint)

    def tick(self):
        latest = self.latest

        if latest is not None:
            frame, gesture, result, hint = latest
            self.latest = None

            # The first frame is the camera coming alive: clear the
            # "starting" line the moment there is a picture to show.
            if not self.first_frame_seen:
                self.first_frame_seen = True
                if not self.engine.config.dry_run:
                    self.result_label.configure(text="show a hand",
                                                fg=ACCENT)

            shown = (gesture or "--").replace("_", " ")
            self.gesture_label.configure(text=shown)

            if result is not None:
                said = result.detail if result.ok else (result.error or "")
                self.result_label.configure(
                    text=said or str(result),
                    fg=ACCENT if result.ok else "#e06c75")
            elif self.engine.config.dry_run:
                # Paused is not dead, and it should never look dead.
                self.result_label.configure(
                    text="paused -- watching, touching nothing", fg=DIM)

            self.hint_label.configure(text=hint or "")

            self.beat += 1

            if self.engine.config.show_preview:
                try:
                    import cv2
                    from PIL import Image, ImageTk

                    small = cv2.resize(frame, PREVIEW_SIZE)
                    image = Image.fromarray(
                        cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                    self.preview_photo = ImageTk.PhotoImage(image)
                    self.preview.configure(image=self.preview_photo,
                                           width=PREVIEW_SIZE[0],
                                           height=PREVIEW_SIZE[1])
                except Exception:
                    pass
            else:
                # The heartbeat: this line only runs when a frame
                # actually arrived, so a beating dot IS the camera
                # working -- and a frozen dot is the truth about that
                # too.
                dot = "●" if (self.beat // 12) % 2 else "○"
                self.preview.configure(
                    text=f"{dot} watching -- picture hidden "
                         f"(click to show)")

        if not self.stop.is_set():
            self.root.after(40, self.tick)

    def run(self):
        from integration import runner
        from vision import Camera, CameraError

        # The camera is opened here, on the main thread, before the
        # vision worker starts -- AVFoundation acquires it through the
        # main run loop, and a camera opened from the worker (under the
        # window's mainloop) hangs waiting for a run loop that never
        # comes.  This was the two-minute freeze.  A failure to open
        # leaves the window standing with a plain reason on it, rather
        # than an empty crash.
        camera = None
        far = getattr(self.args, "far", False)

        try:
            camera = Camera(self.args.camera,
                            width=1600 if far else 640,
                            height=1200 if far else 480).open()
        except CameraError as exc:
            self.result_label.configure(text=f"no camera: {exc}",
                                        fg="#e06c75")

        def vision():
            runner.run(self.engine, self.args,
                       on_frame=self.take_frame,
                       should_stop=self.stop.is_set,
                       camera=camera)

        if camera is not None:
            self.worker = threading.Thread(target=vision, daemon=True)
            self.worker.start()

        self.root.after(40, self.tick)
        self.root.mainloop()

        self.stop.set()

        if self.worker is not None:
            self.worker.join(timeout=3.0)

        return 0
