"""The misfire number must be trustworthy, or it will steer the launch wrong.

find_reversals() is a heuristic, so these tests pin its edges: what
counts as taking a command back, what counts as leaning on a gesture on
purpose, and which traffic is never judged at all.  The log loader is
tested against the file as it really is -- appended by a path that must
never fail, so half-written lines are normal, not exceptional.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import report


def event(command, t, status="OK", source="gesture"):
    stamp = datetime.fromtimestamp(1_700_000_000 + t, tz=timezone.utc)
    return {"command": command, "status": status, "source": source,
            "timestamp": stamp.isoformat(), "t": 1_700_000_000 + t}


class TestReversals(unittest.TestCase):
    def test_quick_opposite_is_a_misfire(self):
        pairs = report.find_reversals([event("VOLUME_UP", 0),
                                       event("VOLUME_DOWN", 1.0)])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0]["command"], "VOLUME_UP")

    def test_slow_opposite_is_a_change_of_mind(self):
        pairs = report.find_reversals([event("VOLUME_UP", 0),
                                       event("VOLUME_DOWN", 10.0)])
        self.assertEqual(pairs, [])

    def test_the_undoer_is_not_a_candidate(self):
        """UP, DOWN, DOWN is one misfire, not two.

        The middle DOWN took the UP back; counting it again as a command
        that the last DOWN then 'undid' would double every correction.
        """

        pairs = report.find_reversals([event("VOLUME_UP", 0),
                                       event("VOLUME_DOWN", 1.0),
                                       event("VOLUME_DOWN", 2.0)])
        self.assertEqual(len(pairs), 1)

    def test_leaning_on_a_gesture_is_not_a_misfire(self):
        """UP, UP, DOWN: the first UP was plainly meant.

        A repeat of the same command ends the search for that event --
        two deliberate notches up followed by a correction blames only
        the second notch.
        """

        pairs = report.find_reversals([event("VOLUME_UP", 0),
                                       event("VOLUME_UP", 1.0),
                                       event("VOLUME_DOWN", 2.0)])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0]["t"], event("", 1.0)["t"])

    def test_play_pause_toggled_straight_back(self):
        """PLAY_PAUSE is its own opposite, and the repeat rule must not
        swallow it -- a toggle-and-back is exactly the misfire shape."""

        pairs = report.find_reversals([event("PLAY_PAUSE", 0),
                                       event("PLAY_PAUSE", 1.5)])
        self.assertEqual(len(pairs), 1)

    def test_different_axes_do_not_pair(self):
        pairs = report.find_reversals([event("VOLUME_UP", 0),
                                       event("BRIGHTNESS_DOWN", 1.0)])
        self.assertEqual(pairs, [])


class TestWhatIsJudged(unittest.TestCase):
    def test_only_performed_commands_count(self):
        """THROTTLED and ERROR moved nothing, so nothing was taken back."""

        events = [event("VOLUME_UP", 0, status="THROTTLED"),
                  event("VOLUME_DOWN", 1.0, status="ERROR")]
        self.assertEqual(report.performed(events), [])

    def test_deliberate_sources_are_never_judged(self):
        """A hotkey correction is a person at a keyboard, not the camera."""

        events = [event("VOLUME_UP", 0, source="hotkey"),
                  event("VOLUME_DOWN", 1.0, source="hotkey"),
                  event("VOLUME_UP", 2.0, source="selftest"),
                  event("VOLUME_UP", 3.0, source="simulator"),
                  event("VOLUME_UP", 4.0, source="cli")]
        self.assertEqual(report.from_camera(events), [])

    def test_untagged_lines_are_judged_as_camera(self):
        """Logs written before tagging existed are still evidence."""

        events = [event("VOLUME_UP", 0, source="")]
        self.assertEqual(len(report.from_camera(events)), 1)


class TestSessions(unittest.TestCase):
    def test_a_long_gap_splits_two_sessions(self):
        spans = report.sessions([event("VOLUME_UP", 0),
                                 event("VOLUME_UP", 60),
                                 event("VOLUME_UP", 60 + 3600)])
        self.assertEqual(len(spans), 2)


class TestLoader(unittest.TestCase):
    def test_broken_lines_are_skipped_not_fatal(self):
        path = Path(tempfile.mkdtemp()) / "commands.jsonl"
        good = event("VOLUME_UP", 0)
        good.pop("t")
        path.write_text(json.dumps(good) + "\n"
                        + "{half a line\n"
                        + json.dumps({"no": "command here"}) + "\n")

        events = report.load_events(path)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["command"], "VOLUME_UP")


class TestRender(unittest.TestCase):
    def test_the_verdict_leads_and_the_numbers_agree(self):
        events = [event("VOLUME_UP", 0),
                  event("VOLUME_DOWN", 1.0),
                  event("PLAY_PAUSE", 5.0),
                  event("VOLUME_UP", 6.0, status="THROTTLED")]

        text = report.render(events, "commands.jsonl")

        self.assertIn("1 of 3 camera commands", text)
        self.assertIn("VOLUME_UP", text)

    def test_a_clean_log_says_so(self):
        events = [event("VOLUME_UP", 0), event("PLAY_PAUSE", 60.0)]
        text = report.render(events, "commands.jsonl")
        self.assertIn("No misfires suspected", text)

    def test_a_short_log_gets_no_rate(self):
        """Minutes of use cannot honestly become a misfires-per-hour."""

        events = [event("VOLUME_UP", 0), event("VOLUME_DOWN", 1.0)]
        text = report.render(events, "commands.jsonl")
        self.assertNotIn("every", text)

    def test_days_are_kept_apart(self):
        """The tuning loop's instrument: each day judges its own change.

        One rate across the whole log blurs every threshold change
        together; the day table is what says whether yesterday's tweak
        cut the misfires or not.
        """

        day = 24 * 3600
        events = [event("VOLUME_UP", 0), event("VOLUME_DOWN", 1.0),
                  event("PLAY_PAUSE", 2 * day)]

        rows = report.by_day(report.from_camera(report.performed(events)),
                             report.find_reversals(
                                 report.from_camera(report.performed(events))))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1], [2, 1], "two commands, one misfire")
        self.assertEqual(rows[1][1], [1, 0])

    def test_a_long_log_gets_one(self):
        """40 min of use would split at the 30-min session gap, so the
        stretch stays inside one session: 25 min, quiet but continuous."""

        events = [event("VOLUME_UP", 0), event("VOLUME_DOWN", 1.0),
                  event("PLAY_PAUSE", 25 * 60)]
        text = report.render(events, "commands.jsonl")
        self.assertIn("one misfire every", text)


if __name__ == "__main__":
    unittest.main()
