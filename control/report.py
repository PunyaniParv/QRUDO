"""The reliability instrument: what the command log was kept for.

Every command QRUDO performs is a line in logs/commands.jsonl.  This
module reads them back and answers the one question that decides
whether people keep the app installed: how often did a gesture fire
that nobody meant?

Nobody files a bug for a wrong volume nudge.  They take it back -- and
the taking-back is itself in the log, as a command followed almost at
once by its opposite.  That pair is the misfire signature, so the
false-positive rate can be measured on every machine QRUDO runs on,
forever, with no extra hardware and no questions asked of anyone.

The heuristic is deliberately blunt and it leans one honest way: a
user who genuinely wanted one notch down after one notch up is counted
as a misfire, and a misfire nobody bothered to undo is missed.  The
first inflates, the second deflates, and neither depends on which app
was playing -- so the number is comparable across days and devices,
which is what a launch decision needs.

Only commands that arrived by camera are judged.  A hotkey, the
simulator, the selftest and --command are deliberate by construction,
so they are read for the totals and skipped for the verdict.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

#: A command and the command that takes it back.  PLAY_PAUSE is its own
#: opposite: a toggle nobody meant gets toggled straight back.
OPPOSITES = {
    "VOLUME_UP": "VOLUME_DOWN", "VOLUME_DOWN": "VOLUME_UP",
    "BRIGHTNESS_UP": "BRIGHTNESS_DOWN", "BRIGHTNESS_DOWN": "BRIGHTNESS_UP",
    "REWIND": "FORWARD", "FORWARD": "REWIND",
    "TARGET_NEXT": "TARGET_PREV", "TARGET_PREV": "TARGET_NEXT",
    "PLAY_PAUSE": "PLAY_PAUSE",
}

#: An opposite this many seconds after a command reads as taking it back.
#: Longer than the 0.6 s cooldown, shorter than a change of mind.
REVERSAL_SECONDS = 3.0

#: A quiet gap this long splits two sessions, so hours in front of the
#: camera are not inflated by lunch.
SESSION_GAP_SECONDS = 30 * 60

#: Routes where a human pressed a key on purpose.  Their commands count
#: in the totals but never toward the misfire verdict.
DELIBERATE_SOURCES = {"hotkey", "simulator", "selftest", "cli"}


def run(config, path: str | Path | None = None) -> int:
    """Print the reliability report.  The CLI lands here for --report."""
    from paths import resolve

    path = Path(path) if path else resolve(config.log_dir) / "commands.jsonl"
    if not path.exists():
        print(f"  no command log at {path} yet -- run QRUDO, use it a"
              f"\n  while, and come back.")
        return 1

    events = load_events(path)
    if not events:
        print(f"  {path} holds no readable commands yet.")
        return 1

    print(render(events, path))
    return 0


def load_events(path: Path) -> list[dict]:
    """The log lines that describe a command, oldest first.

    The file is appended by a path that must never fail, so a half
    line from a crash or a stray non-command object is normal; anything
    unreadable is skipped, not fatal.
    """
    events = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
                event["t"] = datetime.fromisoformat(event["timestamp"]).timestamp()
            except (ValueError, KeyError, TypeError):
                continue
            if "command" in event and "status" in event:
                events.append(event)
    events.sort(key=lambda e: e["t"])
    return events


def performed(events) -> list[dict]:
    """Commands that actually changed the machine: OK, and nothing else.

    NOOP moved nothing, THROTTLED was absorbed, ERROR failed -- none of
    them is something a user would take back.
    """
    return [e for e in events if e["status"] == "OK"]


def from_camera(events) -> list[dict]:
    """The commands the verdict may judge.

    An untagged source is an old log written before tagging existed;
    those lines are almost all camera traffic, so they are judged too
    rather than thrown away -- render() says when that happened.
    """
    return [e for e in events if e.get("source", "") not in DELIBERATE_SOURCES]


def find_reversals(commands, window: float = REVERSAL_SECONDS):
    """Pairs of (misfire, the command that took it back).

    For each command, look ahead no further than ``window`` seconds for
    the first event on the same axis.  If it is the opposite, the first
    of the pair is a suspected misfire and its undoer is consumed --
    a correction is deliberate, so it must not become a candidate
    itself.  If it is the same command again, the user was leaning on
    the gesture on purpose, and the search ends there.
    """
    suspected = []
    consumed = set()

    for i, event in enumerate(commands):
        if i in consumed:
            continue
        counter = OPPOSITES.get(event["command"])
        if counter is None:
            continue
        for j in range(i + 1, len(commands)):
            if j in consumed:
                continue
            later = commands[j]
            if later["t"] - event["t"] > window:
                break
            if later["command"] == event["command"] and counter != event["command"]:
                break
            if later["command"] == counter:
                suspected.append((event, later))
                consumed.add(j)
                break

    return suspected


def sessions(events, gap: float = SESSION_GAP_SECONDS):
    """(first, last) timestamps of each stretch of continuous use."""
    spans = []
    for event in events:
        if spans and event["t"] - spans[-1][1] <= gap:
            spans[-1][1] = event["t"]
        else:
            spans.append([event["t"], event["t"]])
    return spans


def by_day(camera, misfires):
    """Camera commands and misfires per calendar day, oldest first.

    This is the tuning loop's instrument: change one threshold, use
    QRUDO for a day, and the day answers whether the change cut.  Rates
    across whole logs blur every change together; days keep them apart.
    """
    fired = {id(event) for event, _undone_by in misfires}
    days: dict[str, list[int]] = {}
    for event in camera:
        day = datetime.fromtimestamp(event["t"]).strftime("%Y-%m-%d")
        row = days.setdefault(day, [0, 0])
        row[0] += 1
        if id(event) in fired:
            row[1] += 1
    return sorted(days.items())


def render(events, path) -> str:
    """The report itself, plain text, worst news first."""
    done = performed(events)
    camera = from_camera(done)
    misfires = find_reversals(camera)
    spans = sessions(events)
    active_hours = sum(last - first for first, last in spans) / 3600
    untagged = sum(1 for e in done if e.get("source", "") == "")

    lines = ["", f"  QRUDO reliability -- {path}", ""]

    # The verdict, first.  Everything below it is supporting detail.
    if not camera:
        lines += ["  no camera-driven commands in the log yet, so there is",
                  "  no misfire rate to report.", ""]
    elif not misfires:
        lines += [f"  0 of {len(camera)} camera commands were taken back "
                  f"within {REVERSAL_SECONDS:g} s.",
                  "  No misfires suspected.", ""]
    else:
        share = 100 * len(misfires) / len(camera)
        lines += [f"  {len(misfires)} of {len(camera)} camera commands "
                  f"({share:.0f}%) were taken back within "
                  f"{REVERSAL_SECONDS:g} s -- suspected misfires."]
        # A rate needs time behind it; on a minutes-long log the share
        # above is the only honest number.
        if active_hours * 60 >= 10:
            minutes = (active_hours * 60) / len(misfires)
            lines += [f"  That is one misfire every {minutes:.0f} minute(s) "
                      f"in front of the camera."]
        lines += [""]

        by_command: dict[str, int] = {}
        for fired, _undone_by in misfires:
            by_command[fired["command"]] = by_command.get(fired["command"], 0) + 1
        for name, count in sorted(by_command.items(), key=lambda kv: -kv[1]):
            lines += [f"    {name:<16} {count:>4} taken back"]
        lines += [""]

    hours = f"{active_hours:.1f}" if active_hours >= 0.1 else "under 0.1"
    lines += [f"  {len(done)} commands performed over {len(spans)} "
              f"session(s), {hours} h of use.", ""]

    counts: dict[str, list[int]] = {}
    for event in events:
        row = counts.setdefault(event["command"], [0, 0, 0])
        if event["status"] == "OK":
            row[0] += 1
        elif event["status"] == "THROTTLED":
            row[1] += 1
        elif event["status"] in ("ERROR", "UNSUPPORTED"):
            row[2] += 1

    lines += [f"    {'command':<16} {'ok':>5} {'throttled':>10} {'failed':>7}"]
    for name, (ok, throttled, failed) in sorted(counts.items(),
                                                key=lambda kv: -kv[1][0]):
        if name == "NONE":
            continue
        lines += [f"    {name:<16} {ok:>5} {throttled:>10} {failed:>7}"]
    lines += [""]

    days = by_day(camera, misfires)
    if len(days) >= 2:
        lines += [f"    {'day':<12} {'commands':>9} {'misfires':>9}"]
        for day, (count, taken_back) in days:
            share = f"{100 * taken_back / count:.0f}%" if count else "-"
            lines += [f"    {day:<12} {count:>9} {taken_back:>6} {share:>5}"]
        lines += ["",
                  "  Tune one thing at a time, and let the next day's row",
                  "  say whether it cut.", ""]
    elif camera:
        lines += ["  A day-by-day trend appears here once the log spans",
                  "  more than one day -- that is how a tuning change is",
                  "  judged: against the days after it.", ""]

    if untagged:
        lines += [f"  {untagged} command(s) predate source tagging and were",
                  "  judged as camera traffic.  Hotkey, simulator, selftest",
                  "  and --command traffic is otherwise never judged.", ""]

    return "\n".join(lines)
