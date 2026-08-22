"""Specification/contract tests for docs/QRUDO_INTELLIGENCE_V1.md.

These tests enshrine the *engineering invariants* (I1-I9) of the v1 spec as
executable contracts against the code that exists today.  They change no
production behaviour: they only read the current architecture and assert that
the invariants the spec demands are already satisfied (or, for the still-
proposed context capsule, that the proposed shape is sound).

They deliberately depend only on the deterministic, offline layers
(voice/bridge, ai/tools, ai/schema) and never require hardware or LLM SDKs,
so the suite stays green on every machine.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.config import AIConfig
from ai.memory import Memory, NullMemory
from ai.provider import NullProvider
from ai.schema import Message, ToolCall
from ai.tools.registry import (
    ToolRegistry,
    ensure_prebuilt_tools,
    get_registry,
    reset_registry,
)
from control import Command, ControlEngine, ControlConfig
from control.backends.null import NullController


def reset_and_ensure() -> ToolRegistry:
    """Fresh, well-formed registry for each test (same pattern as the AI tests)."""
    reset_registry()
    return ensure_prebuilt_tools(get_registry())


# ---------------------------------------------------------------------------
# I1 - Fast path is deterministic and LLM-free
# ---------------------------------------------------------------------------
class TestFastPathDeterministic(unittest.TestCase):
    """Representative built-in phrases must resolve on the fast path (no LLM)."""

    PHRASES = [
        "increase the volume",
        "volume up",
        "make it louder",
        "turn the volume down",
        "make it quieter",
        "play pause",
        "rewind",
        "forward",
        "brightness up",
        "brightness down",
        "switch the target",
        "previous target",
    ]

    def test_builtin_phrases_resolve_to_a_command(self):
        from voice.bridge import VoiceIntentRouter

        router = VoiceIntentRouter()
        for phrase in self.PHRASES:
            with self.subTest(phrase=phrase):
                self.assertIsNotNone(
                    router.route(phrase),
                    f"'{phrase}' must resolve on the deterministic fast path",
                )
                self.assertIsNotNone(
                    router.classify(phrase),
                    f"'{phrase}' must classify to a Command (fast path)",
                )

    def test_unmatched_request_returns_none(self):
        """A request with no supported action routes to None -- the ONLY
        condition under which the Assistant (LLM) may be consulted."""
        from voice.bridge import VoiceIntentRouter

        router = VoiceIntentRouter()
        self.assertIsNone(router.route("search the web for clouds"))
        self.assertIsNone(router.classify("what is the weather like"))


class TestFastPathNeverImportsAI(unittest.TestCase):
    """The deterministic routing modules must stay opaque to the AI package."""

    FORBIDDEN = ("ai", "assistant", "provider")

    def test_bridge_and_commands_do_not_reference_ai(self):
        for rel in ("voice/bridge.py", "control/commands.py"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            for token in self.FORBIDDEN:
                with self.subTest(file=rel, token=token):
                    self.assertNotIn(
                        f"import {token}", text,
                        f"{rel} must not import the AI package ({token})",
                    )
                    self.assertNotIn(
                        f"from {token}", text,
                        f"{rel} must not import the AI package ({token})",
                    )

    def test_router_has_no_provider_handle(self):
        from voice.bridge import VoiceIntentRouter

        router = VoiceIntentRouter()
        self.assertFalse(hasattr(router, "provider"))
        self.assertFalse(hasattr(router, "respond"))


# ---------------------------------------------------------------------------
# I2 - Single OS boundary: ToolRegistry is the only bridge
# ---------------------------------------------------------------------------
class TestSingleBoundary(unittest.TestCase):
    def test_manifest_is_whitelist_only(self):
        reg = reset_and_ensure()
        names = set(reg._tools.keys())

        # The closed, safe set we expect today.
        allowed = {
            "volume_up", "volume_down", "brightness_up", "brightness_down",
            "play_pause", "rewind", "forward", "target_next", "target_prev",
            "catalog_action", "custom_action",
            "capabilities", "current_target", "available_commands",
        }
        self.assertTrue(allowed.issubset(names))

        # Anything that would reach outside the safety boundary is forbidden.
        forbidden = {"shell", "shell_exec", "exec", "run_python", "eval",
                     "read_file", "write_file", "list_dir", "http_get",
                     "network", "web_search", "screenshot", "process_list"}
        self.assertFalse(forbidden.intersection(names))

    def test_unknown_tool_rejected(self):
        reg = reset_and_ensure()
        engine = ControlEngine(controller=NullController(ControlConfig()),
                               config=ControlConfig())
        tc = ToolCall(id="c1", name="rm_rf_all", arguments={})
        result = reg.call(engine, tc)
        self.assertFalse(result.success)
        self.assertIn("not whitelisted", (result.error or "").lower())


# ---------------------------------------------------------------------------
# I3 / I4 - Provider-neutrality and strict validation
# ---------------------------------------------------------------------------
class TestProviderNeutrality(unittest.TestCase):
    def test_null_provider_is_noop(self):
        provider = NullProvider()
        self.assertFalse(provider.available())
        turn = provider.respond([Message(role="user", content="hi")], [])
        self.assertEqual(turn.tool_calls, [])
        self.assertEqual(turn.message.content, "AI is currently unavailable")

    def test_default_memory_is_readonly_noop(self):
        mem: Memory = NullMemory()
        self.assertEqual(mem.get_recent(5), [])
        mem.remember("user", "anything")  # must not raise
        mem.clear()  # must not raise
        self.assertEqual(mem.get_recent(5), [])

    def test_ai_import_does_not_load_llm_sdk(self):
        code = (
            "import sys; "
            "import ai, ai.config, ai.memory, ai.provider, ai.schema; "
            "bad = [m for m in ('openai','langchain','anthropic',"
            "'google.generativeai','requests','httpx','ollama') "
            "if m in sys.modules]; "
            "assert not bad, f'LLM/network SDKs loaded: {bad}'; "
            "print('AI_LAYER_OK')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("AI_LAYER_OK", result.stdout)

    def test_malformed_arguments_rejected(self):
        reg = reset_and_ensure()
        engine = ControlEngine(controller=NullController(ControlConfig()),
                               config=ControlConfig())
        # volume_up schema has no 'nonsense' key -> unexpected argument.
        tc = ToolCall(id="c1", name="volume_up", arguments={"nonsense": 1})
        result = reg.call(engine, tc)
        self.assertFalse(result.success)
        self.assertIn("malformed", (result.error or "").lower())

    def test_catalog_action_requires_job_name(self):
        reg = reset_and_ensure()
        engine = ControlEngine(controller=NullController(ControlConfig()),
                               config=ControlConfig())
        tc = ToolCall(id="c1", name="catalog_action", arguments={})
        result = reg.call(engine, tc)
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# I3 - AI disabled by default (zero behaviour change at import/runtime)
# ---------------------------------------------------------------------------
class TestAIDisabledByDefault(unittest.TestCase):
    def test_ai_disabled_default(self):
        self.assertFalse(AIConfig.load().ai_enabled)


# ---------------------------------------------------------------------------
# I9 - Proposed RequestContext / Response capsule must be JSON-serialisable
# ---------------------------------------------------------------------------
class TestProposedCapsule(unittest.TestCase):
    """The §15 context/response structures are a planned provider boundary;
    they must remain plain-JSON so they survive any future process/socket
    split, exactly like Command strings do today."""

    def _request_context(self) -> dict:
        return {
            "schema": "qrudo/request_context/v1",
            "turn_id": "a17f4c92-0000-4000-8000-000000000000",
            "ts": "2026-08-21T12:00:00Z",
            "modality": "voice",
            "user": {"text": "turn it up", "normalized": "turn it up",
                     "language": "en", "wake": "hey qrudo"},
            "routing": {"deterministic_command": None, "catalog_job": None,
                        "route_confidence": 0.0, "decision": "conversational"},
            "state": {"likely_affect": "neutral", "affect_confidence": 0.0,
                      "evidence": []},
            "capabilities": ["volume_control", "brightness_control"],
            "target": {"current": "auto"},
            "memory": {"enabled": False, "recent": [], "durable": []},
        }

    def test_request_context_round_trips(self):
        ctx = self._request_context()
        restored = json.loads(json.dumps(ctx))
        self.assertEqual(restored, ctx)
        for key in ("schema", "turn_id", "modality", "user", "routing",
                    "state", "capabilities", "target", "memory"):
            self.assertIn(key, ctx)

    def test_response_round_trips(self):
        resp = {
            "schema": "qrudo/response/v1",
            "turn_id": "a17f4c92-0000-4000-8000-000000000000",
            "text": "Volume's up.",
            "spoken": "Volume's up.",
            "tone": "brief",
            "tool_calls": [],
            "confirm_required": False,
            "behaviour": "action",
        }
        self.assertEqual(json.loads(json.dumps(resp)), resp)


if __name__ == "__main__":
    unittest.main(verbosity=2)

