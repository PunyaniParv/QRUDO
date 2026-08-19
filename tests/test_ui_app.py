"""The application shell, tested where a headless suite can reach.

The window itself needs a display; what must not regress without one
is the settings plumbing -- typed values landing on the live config
safely -- and the promise that the shell rides the same runner loop
via its two hooks rather than owning a loop of its own.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.config import ControlConfig
from ui.app import apply_settings


class TestApplySettings(unittest.TestCase):
    def test_numbers_land_typed(self):
        config = ControlConfig()

        changed = apply_settings(config, {"volume_step": "15",
                                          "cooldown_seconds": "0.8"})

        self.assertEqual(config.volume_step, 15)
        self.assertEqual(config.cooldown_seconds, 0.8)
        self.assertEqual(sorted(changed),
                         ["cooldown_seconds", "volume_step"])

    def test_garbage_leaves_the_old_value_standing(self):
        """A save must never crash or corrupt over a typo."""

        config = ControlConfig()
        before = config.volume_step

        changed = apply_settings(config, {"volume_step": "loud"})

        self.assertEqual(config.volume_step, before)
        self.assertEqual(changed, [])

    def test_unchanged_values_do_not_report_as_changes(self):
        config = ControlConfig()

        changed = apply_settings(config,
                                 {"volume_step": str(config.volume_step)})

        self.assertEqual(changed, [])

    def test_unknown_names_are_ignored(self):
        """The page and the config may drift; drift must be harmless."""

        config = ControlConfig()

        self.assertEqual(apply_settings(config, {"warp_speed": "9"}), [])

    def test_strings_are_stripped(self):
        config = ControlConfig()

        apply_settings(config, {"target_app": "  Spotify  "})

        self.assertEqual(config.target_app, "Spotify")


class TestTheShellRidesTheRunner(unittest.TestCase):
    def test_the_runner_offers_the_two_hooks(self):
        """on_frame and should_stop are the whole contract between the
        window and the loop; losing either quietly forks the app into
        two camera loops with two behaviours."""

        from integration import runner

        parameters = inspect.signature(runner.run).parameters

        self.assertIn("on_frame", parameters)
        self.assertIn("should_stop", parameters)

    def test_the_runner_accepts_a_preopened_camera(self):
        """The two-minute freeze, guarded.

        AVFoundation acquires the camera through the main run loop, so
        in the packaged app the window must open it on the main thread
        and hand it in -- a camera opened on the vision worker (under
        the toolkit's mainloop) waits on a run loop that never comes.
        The runner must keep accepting one already open, or the window
        is forced back to opening it on the worker, which hangs.
        """

        from integration import runner

        self.assertIn("camera", inspect.signature(runner.run).parameters)

    def test_the_window_opens_the_camera_before_the_worker(self):
        """It must open on the main thread -- in _start_vision, which
        run() calls directly (not inside the vision() thread it spawns)
        -- so the source names Camera ahead of the worker start."""

        import inspect as _inspect

        from ui.app import App

        source = _inspect.getsource(App._start_vision)
        camera_at = source.find("Camera(")
        worker_at = source.find("Thread(target=vision")

        self.assertNotEqual(camera_at, -1,
                            "_start_vision must open the camera")
        self.assertNotEqual(worker_at, -1,
                            "_start_vision must start the worker")
        self.assertLess(camera_at, worker_at,
                        "the camera must open before the worker starts")

        # And run() must call _start_vision on the main thread, not in a
        # worker -- the whole point of the AVFoundation fix.
        run_source = _inspect.getsource(App.run)
        self.assertIn("_start_vision", run_source)

    def test_starting_the_camera_never_asks_it_to_stop(self):
        """The two-second death, pinned.

        A bad edit once left quit()'s shutdown lines on the tail of
        _hide_retry -- which _start_vision calls on every successful
        camera open.  Every launch then set the stop event before its
        own vision loop drew breath: the camera ran two seconds, the
        loop saw a requested stop and ended cleanly, and nothing was
        logged, because clean stops are not errors.  Days of 'the
        camera turns off by itself' traced back to these lines.

        So: nothing on the path that STARTS vision may touch the stop
        event or join the worker.  Only quit(), and run() after its
        mainloop has already ended, may.
        """

        import inspect as _inspect

        from ui.app import App

        for starter in (App._start_vision, App._hide_retry,
                        App._show_retry, App._camera_died,
                        App.take_frame):
            source = _inspect.getsource(starter)

            self.assertNotIn("stop.set", source,
                             f"{starter.__name__} must never set the "
                             f"stop event -- that was the silent "
                             f"two-second camera death")
            self.assertNotIn("worker.join", source,
                             f"{starter.__name__} must not join the "
                             f"vision worker")

        # And the epilogue belongs to run(), after mainloop: the worker
        # must not outlive the window.
        run_source = _inspect.getsource(App.run)
        self.assertIn("mainloop", run_source)
        self.assertIn("stop.set", run_source)

    def test_the_pulse_reschedules_itself_even_when_a_beat_fails(self):
        """An exception escaping a root.after callback ends the chain
        silently: Tk prints to a stderr nobody sees and never runs the
        callback again.  For tick that is the window freezing and the
        dead-camera watchdog going blind; for the recorder's poll it is
        'hold still' freezing mid-recording.  Both must book their next
        beat in a finally, so one bad frame cannot end the pulse."""

        import inspect as _inspect

        from ui.app import App, Recorder

        for pulse in (App.tick, Recorder._poll):
            source = _inspect.getsource(pulse)

            self.assertIn("finally", source,
                          f"{pulse.__qualname__} must reschedule in a "
                          f"finally, or one exception silently ends it")
            self.assertIn(".after(", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
