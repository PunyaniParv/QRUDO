"""The command interface (spec section A).

``ControlEngine.execute(command)`` is the single entry point the Vision Engine
calls.  It never raises and never blocks for long: every failure comes back as
a ``CommandResult`` with ``status == ERROR``, because a missing audio device
must not kill the camera loop (spec F).

Platform specifics live in ``qrudo.backends``; nothing in this file knows what
an osascript is.
"""

from __future__ import annotations

import platform
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import log
from .commands import ACTIONABLE_COMMANDS, Command, Status
from .config import ControlConfig

#: Sentinel pushed onto the worker queue to stop it.
_STOP = object()


class ControlError(Exception):
    """Raised by a backend when an OS action fails.  Caught by ControlEngine."""


class UnsupportedCommand(ControlError):
    """The backend cannot perform this command on this machine at all."""


@dataclass(frozen=True)
class CommandResult:
    command: str
    status: str
    detail: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    #: The route the command arrived by -- "gesture", "hotkey", "simulator",
    #: "selftest", "cli".  It travels into the event log so the reliability
    #: report can judge the camera by the camera's own commands alone.
    source: str = ""

    @property
    def ok(self) -> bool:
        """True when the OS action was performed (or was a deliberate no-op)."""
        return self.status in (Status.OK, Status.NOOP)

    def __str__(self) -> str:
        base = f"{self.command} -> {self.status}"
        if self.detail:
            base += f" ({self.detail})"
        if self.error:
            base += f" [{self.error}]"
        return base


class Controller(ABC):
    """What a platform backend must provide.

    Each method performs one OS action and returns a short human-readable
    description of what happened ("volume 60 -> 65").  Failures raise
    ``ControlError``; the engine turns those into results.
    """

    name = "abstract"

    @abstractmethod
    def volume_up(self, step: int) -> str: ...

    @abstractmethod
    def volume_down(self, step: int) -> str: ...

    @abstractmethod
    def play_pause(self) -> str: ...

    @abstractmethod
    def rewind(self, seconds: int) -> str: ...

    @abstractmethod
    def forward(self, seconds: int) -> str: ...

    @abstractmethod
    def brightness_up(self, step: int) -> str: ...

    @abstractmethod
    def brightness_down(self, step: int) -> str: ...

    def preflight(self) -> list[str]:
        """Return human-readable warnings about this machine (missing
        permissions, unsupported hardware).  An empty list means all good."""
        return []

    def read_state(self) -> dict[str, float]:
        """Whatever this platform can measure, as 0.0-1.0 fractions.

        Keys are free-form ("volume", "brightness", "muted").  The self-test
        prints this before and after, and uses it to put the machine back
        exactly -- undoing a command with its opposite is not enough near the
        ends of a scale, where the command clamps but its opposite does not.
        Return {} for platforms that cannot read anything back.
        """
        return {}

    def restore_state(self, state: dict[str, float]) -> None:
        """Put back what :meth:`read_state` measured.  Best effort."""
        return None


class ControlEngine:
    """Dispatch, debounce, log.  This is the object the rest of QRUDO holds."""

    def __init__(self, controller: Controller | None = None,
                 config: ControlConfig | None = None,
                 on_result: Callable[[CommandResult], None] | None = None) -> None:
        self.config = config or ControlConfig.load()
        self.controller = controller or get_controller(self.config)
        self.on_result = on_result
        self.log = log.get_logger("engine")
        self._last_run: dict[str, float] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._worker: threading.Thread | None = None
        # Each area declares the commands it owns, so adding one means
        # editing that area rather than this table.
        from . import brightness, media, targets, volume

        self._handlers = {}

        for area in (volume, media, brightness):
            self._handlers.update(area.handlers(self.controller, self.config))

        # The target commands are engine-level, not OS actions: they move
        # where the other commands go, so no backend has a method for them.
        self.targets = targets.TargetResolver(self.config)
        self._handlers.update(targets.handlers(self.targets))
        # Fail loudly at construction if a command has no handler, rather than
        # at demo time.
        missing = [c for c in ACTIONABLE_COMMANDS if c not in self._handlers]
        if missing:
            raise RuntimeError(f"no handler registered for: {missing}")

    # -- public API ---------------------------------------------------------

    def execute(self, command: Command | str, *, force: bool = False,
                source: str = "") -> CommandResult:
        """Run one command.  Never raises.

        ``force=True`` bypasses the cooldown (used by the simulator, where every
        keypress is a deliberate human action).  ``source`` names the route the
        command arrived by; it is carried into the event log, nothing more.
        """
        try:
            command = Command(command) if not isinstance(command, Command) else command
        except ValueError:
            return self._finish(CommandResult(
                command=str(command), status=Status.ERROR,
                error=f"unknown command {command!r}", source=source))

        if command is Command.NONE:
            return self._finish(CommandResult(
                command=command.value, status=Status.NOOP, source=source))

        now = time.monotonic()
        if not force and (now - self._last_run.get(command.value, -1e9)) < self.config.cooldown_seconds:
            return self._finish(CommandResult(
                command=command.value, status=Status.THROTTLED,
                detail=f"within {self.config.cooldown_seconds:g}s cooldown",
                source=source))

        if self.config.dry_run:
            self._last_run[command.value] = now
            return self._finish(CommandResult(
                command=command.value, status=Status.OK,
                detail="dry-run, OS untouched", source=source))

        started = time.perf_counter()
        try:
            detail = self._handlers[command]()
            status, error = Status.OK, None
        except UnsupportedCommand as exc:
            detail, status, error = "", Status.UNSUPPORTED, str(exc)
        except Exception as exc:  # a backend bug must not kill the camera loop
            detail, status, error = "", Status.ERROR, f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter() - started) * 1000

        if status is Status.OK:
            self._last_run[command.value] = now

        return self._finish(CommandResult(
            command=command.value, status=status, detail=detail or "",
            error=error, duration_ms=round(elapsed, 1), source=source))

    def submit(self, command: Command | str, source: str = "") -> None:
        """Non-blocking version of :meth:`execute` for the camera loop.

        Volume changes cost ~200 ms because macOS makes us shell out to
        ``osascript``; at 30 fps that is six dropped frames.  ``submit()`` hands
        the command to a single worker thread and returns immediately, so the
        preview stays smooth.

        Results still go to the log, and to ``on_result`` if one was set.
        Commands are executed strictly in order, one at a time.
        """
        if self._worker is None:
            self._start_worker()
        try:
            self._queue.put_nowait((command, source))
        except queue.Full:
            # The backlog is already several commands deep; dropping is the
            # right call -- a stale volume nudge helps nobody.
            self.log.warning("command queue full, dropped %s", command)

    def close(self, timeout: float = 2.0) -> None:
        """Stop the worker threads.  Safe to call even if none started."""
        self.targets.stop()
        if self._worker is None:
            return
        self._queue.put(_STOP)
        self._worker.join(timeout)
        self._worker = None

    def __enter__(self) -> "ControlEngine":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def preflight(self) -> list[str]:
        return self.controller.preflight()

    # -- internals ----------------------------------------------------------

    def _start_worker(self) -> None:
        self._worker = threading.Thread(
            target=self._drain, name="qrudo-control", daemon=True)
        self._worker.start()

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            command, source = item
            result = self.execute(command, source=source)
            if self.on_result is not None:
                try:
                    self.on_result(result)
                except Exception as exc:  # a bad callback must not kill the worker
                    self.log.error("on_result callback failed: %s", exc)

    def _finish(self, result: CommandResult) -> CommandResult:
        if result.status is Status.ERROR or result.status is Status.UNSUPPORTED:
            self.log.error("%s", result)
        elif result.status is Status.THROTTLED:
            self.log.debug("%s", result)
        else:
            self.log.info("%s", result)
        log.log_event(result)
        return result


def get_controller(config: ControlConfig | None = None) -> Controller:
    """Pick the backend for this machine."""
    config = config or ControlConfig.load()
    system = platform.system()
    if system == "Darwin":
        from .backends.macos import MacOSController
        return MacOSController(config)
    if system == "Windows":
        from .backends.windows import WindowsController
        return WindowsController(config)
    from .backends.null import NullController
    log.get_logger().warning(
        "no control backend for %s; using NullController (commands are logged only)", system)
    return NullController(config)
