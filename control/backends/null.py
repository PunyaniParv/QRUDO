"""A backend that does nothing but report what it was asked to do.

Used on unsupported platforms and in tests, so the rest of QRUDO can be exercised
on any machine.  Also the template to copy when someone adds Windows or Linux
support: implement the same seven methods.
"""

from __future__ import annotations

from ..config import ControlConfig
from ..executor import Controller


class NullController(Controller):
    name = "null"

    def __init__(self, config: ControlConfig | None = None) -> None:
        self.config = config or ControlConfig()
        self.calls: list[str] = []

    def _record(self, what: str) -> str:
        self.calls.append(what)
        return f"[null] {what}"

    def volume_up(self, step: int) -> str:
        return self._record(f"volume +{step}%")

    def volume_down(self, step: int) -> str:
        return self._record(f"volume -{step}%")

    def play_pause(self) -> str:
        return self._record("play/pause")

    def rewind(self, seconds: int) -> str:
        return self._record(f"rewind {seconds}s")

    def forward(self, seconds: int) -> str:
        return self._record(f"forward {seconds}s")

    def brightness_up(self, step: int) -> str:
        return self._record(f"brightness +{step}%")

    def brightness_down(self, step: int) -> str:
        return self._record(f"brightness -{step}%")

    def preflight(self) -> list[str]:
        return ["NullController active: no OS actions will be performed"]
