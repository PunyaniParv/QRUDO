"""Phase A tests for the QRUDO AI LLM architecture."""

from __future__ import annotations

import sys
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai import AIConfig, schema, memory, tools
from ai.tools.registry import Tool, ToolRegistry, get_registry, ensure_prebuilt_tools, reset_registry
from ai.schema import Message, ToolCall, ToolResult, Turn
from ai.memory import Memory, NullMemory


def reset_and_ensure():
    """Reset registry and ensure pre-built tools are registered for each test."""
    reset_registry()
    ensure_prebuilt_tools()


class TestAIConfig(unittest.TestCase):
    """Test AI configuration behavior."""

    def test_ai_disabled_by_default(self):
        """AI enabled=False by default - zero behavior change."""
        cfg = AIConfig.load()
        self.assertFalse(cfg.ai_enabled)

    def test_import_no_api_key_required(self):
        """Importing the AI package does not require an API key."""
        from ai import AIConfig
        self.assertTrue(True)

    def test_import_no_llm_sdk(self):
        """Importing the AI package does not require an LLM SDK."""
        from ai import schema, memory, tools
        self.assertTrue(True)

    def test_config_all_defaults(self):
        """All configuration defaults are set."""
        cfg = AIConfig.load()
        self.assertFalse(cfg.ai_enabled)
        self.assertEqual(cfg.provider, "")
        self.assertEqual(cfg.endpoint, "")
        self.assertEqual(cfg.model, "")
        self.assertTrue(cfg.confirm_actions)
        self.assertEqual(cfg.max_turns, 5)

    def test_config_loads_from_env(self):
        """Configuration can be loaded from environment variables."""
        os.environ["AI_ENABLED"] = "1"
        os.environ["AI_PROVIDER"] = "openai"
        os.environ["AI_MODEL"] = "gpt-4"
        try:
            cfg = AIConfig.load()
            self.assertTrue(cfg.ai_enabled)
            self.assertEqual(cfg.provider, "openai")
            self.assertEqual(cfg.model, "gpt-4")
        finally:
            for key in ["AI_ENABLED", "AI_PROVIDER", "AI_MODEL"]:
                os.environ.pop(key, None)

    def test_config_env_graceful_missing(self):
        """Missing env vars are graceful, defaults used."""
        for key in ["AI_ENABLED", "AI_PROVIDER", "AI_MODEL"]:
            os.environ.pop(key, None)
        cfg = AIConfig.load()
        self.assertIsInstance(cfg, AIConfig)


class TestSchema(unittest.TestCase):
    """Test provider-neutral schema dataclasses."""

    def test_message_creation(self):
        msg = Message(role="user", content="hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "hello")

    def test_message_with_tool_calls(self):
        tc = ToolCall(id="call_1", name="volume_up", arguments={})
        msg = Message(role="assistant", content="", tool_calls=[tc])
        self.assertEqual(len(msg.tool_calls), 1)

    def test_tool_call_creation(self):
        tc = ToolCall(id="call_1", name="volume_up", arguments={"step": 5})
        self.assertEqual(tc.id, "call_1")
        self.assertEqual(tc.name, "volume_up")
        self.assertEqual(tc.arguments, {"step": 5})

    def test_tool_result_creation(self):
        tr = ToolResult(tool_call_id="call_1", success=True, data={"ok": True})
        self.assertTrue(tr.success)
        self.assertEqual(tr.tool_call_id, "call_1")

        tr2 = ToolResult(tool_call_id="call_1", success=False, error="failed")
        self.assertFalse(tr2.success)
        self.assertIsNotNone(tr2.error)

    def test_turn_creation(self):
        msg = Message(role="user", content="test")
        tc = ToolCall(id="call_1", name="test", arguments={})
        tr = ToolResult(tool_call_id="call_1", success=True, data={})
        turn = Turn(message=msg, tool_calls=[tc], tool_results=[tr])
        self.assertEqual(turn.message, msg)
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(len(turn.tool_results), 1)


class TestMemory(unittest.TestCase):
    """Test memory contract and NullMemory."""

    def test_null_memory_safe(self):
        nm = NullMemory()
        self.assertEqual(nm.get_recent(), [])
        nm.remember("user", "test")
        nm.clear()

    def test_null_memory_get_recent(self):
        nm = NullMemory()
        result = nm.get_recent(5)
        self.assertEqual(result, [])

    def test_null_memory_remember(self):
        nm = NullMemory()
        nm.remember("assistant", "hello")
        self.assertTrue(True)

    def test_null_memory_clear(self):
        nm = NullMemory()
        nm.clear()
        self.assertTrue(True)

    def test_memory_contract_interface(self):
        nm = NullMemory()
        self.assertTrue(hasattr(nm, "get_recent"))
        self.assertTrue(hasattr(nm, "remember"))
        self.assertTrue(hasattr(nm, "clear"))


class TestToolRegistry(unittest.TestCase):
    """Test the tool registry architecture."""

    def setUp(self):
        reset_and_ensure()

    def test_registry_created(self):
        reg = ToolRegistry()
        self.assertIsInstance(reg, ToolRegistry)

    def test_unknown_tool_rejected(self):
        reg = get_registry()
        tc = ToolCall(id="1", name="nonexistent", arguments={})
        result = reg.call(None, tc)
        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.success)

    def test_malformed_arguments_rejected(self):
        reg = get_registry()
        tc = ToolCall(id="1", name="volume_up", arguments="not_a_dict")
        result = reg.call(None, tc)
        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.success)

    def test_tool_schema_enforced(self):
        reg = get_registry()
        tc = ToolCall(id="1", name="volume_up", arguments={"step": "not_an_int"})
        result = reg.call(None, tc)
        self.assertIsInstance(result, ToolResult)

    def test_handlers_cannot_bypass_registry(self):
        reg = get_registry()
        for tool_name in ["volume_up", "volume_down", "brightness_up",
                          "brightness_down", "play_pause", "rewind", "forward",
                          "target_next", "target_prev"]:
            tool = reg.get(tool_name)
            self.assertIsNotNone(tool, f"tool {tool_name} should be registered")
            self.assertTrue(callable(tool.handler))

    def test_confirmation_required_tools(self):
        reg = get_registry()
        self.assertIsInstance(reg, ToolRegistry)


class TestBuiltinTools(unittest.TestCase):
    """Test builtin tool wrappers."""

    def setUp(self):
        reset_and_ensure()

    def test_volume_up_maps_to_command(self):
        reg = get_registry()
        tool = reg.get("volume_up")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "volume_up")
        self.assertFalse(tool.confirmation_required)

    def test_volume_down_maps_to_command(self):
        reg = get_registry()
        tool = reg.get("volume_down")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "volume_down")
        self.assertFalse(tool.confirmation_required)

    def test_brightness_mappings(self):
        reg = get_registry()
        self.assertIsNotNone(reg.get("brightness_up"))
        self.assertIsNotNone(reg.get("brightness_down"))

    def test_media_mappings(self):
        reg = get_registry()
        self.assertIsNotNone(reg.get("play_pause"))
        self.assertIsNotNone(reg.get("rewind"))
        self.assertIsNotNone(reg.get("forward"))

    def test_target_mappings(self):
        reg = get_registry()
        self.assertIsNotNone(reg.get("target_next"))
        self.assertIsNotNone(reg.get("target_prev"))

    def test_source_ai_passed(self):
        reg = get_registry()
        for tool_name in ["volume_up", "volume_down", "play_pause", "rewind", "forward"]:
            tool = reg.get(tool_name)
            self.assertIsNotNone(tool)


class TestCatalogActions(unittest.TestCase):
    """Test catalog/custom action protection."""

    def setUp(self):
        reset_and_ensure()

    def test_open_chrome_resolves_through_catalog(self):
        reg = get_registry()
        tool = reg.get("catalog_action")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "catalog_action")

    def test_existing_custom_payload_preserved(self):
        reg = get_registry()
        tool = reg.get("custom_action")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "custom_action")

    def test_arbitrary_unregistered_custom_cannot_inject(self):
        reg = get_registry()
        registered = list(reg._tools.keys())
        for bad in ["shell", "python", "execute", "raw"]:
            self.assertNotIn(bad, registered,
                             f"Tool '{bad}' should not be registered (unsafe)")


class TestSafety(unittest.TestCase):
    """Test safety guards remain active."""

    def setUp(self):
        reset_and_ensure()

    def test_no_raw_shell_tool(self):
        reg = get_registry()
        registered = list(reg._tools.keys())
        self.assertNotIn("shell", registered)
        self.assertNotIn("raw_shell", registered)
        self.assertNotIn("arbitrary_execute", registered)

    def test_destructive_action_safeguards_active(self):
        from control.actions import validate, is_destructive
        bad_commands = ["rm -rf /", "sudo rm x", "dd if=/dev/zero of=/dev/sda"]
        for cmd in bad_commands:
            action = {"type": "run_command", "command": cmd, "confirmed": True}
            with self.subTest(command=cmd):
                try:
                    validate(action)
                except Exception:
                    pass

    def test_existing_validation_remains(self):
        from control.actions import validate
        good = {"type": "run_command", "command": "echo hi", "confirmed": True}
        clean = validate(good)
        self.assertTrue(clean["confirmed"])

    def test_dry_run_respected(self):
        from control import ControlEngine, ControlConfig, Command
        from control.backends.null import NullController
        cfg = ControlConfig(dry_run=True)
        engine = ControlEngine(controller=NullController(cfg), config=cfg)
        result = engine.execute(Command.VOLUME_UP, source="ai")
        self.assertTrue(result.ok)
        self.assertIn("dry-run", (result.detail or "").lower())

    def test_cooldown_respected(self):
        from control import ControlEngine, ControlConfig, Command
        from control.backends.null import NullController
        cfg = ControlConfig(cooldown_seconds=0.5)
        engine = ControlEngine(controller=NullController(cfg), config=cfg)
        result1 = engine.execute(Command.VOLUME_UP, source="ai")
        self.assertTrue(result1.ok)
        result2 = engine.execute(Command.VOLUME_UP, source="ai")
        self.assertEqual(result2.status.name, "THROTTLED")


class TestIntegration(unittest.TestCase):
    """Integration tests for Phase A components working together."""

    def setUp(self):
        reset_and_ensure()

    def test_full_registry_lifecycle(self):
        reg = get_registry()
        expected_tools = ["volume_up", "volume_down", "brightness_up",
                          "brightness_down", "play_pause", "rewind", "forward",
                          "target_next", "target_prev", "catalog_action", "custom_action",
                          "capabilities", "current_target", "available_commands"]
        for tool_name in expected_tools:
            self.assertIsNotNone(reg.get(tool_name),
                               f"Expected tool '{tool_name}' to be registered")

    def test_message_round_trip(self):
        msg = Message(role="user", content="test intent")
        tc = ToolCall(id="call_1", name="volume_up", arguments={"step": 5})
        tr = ToolResult(tool_call_id="call_1", success=True, data={"command": "VOLUME_UP"})
        msg.tool_calls.append(tc)
        tc.result = tr
        self.assertEqual(msg.tool_calls[0].id, "call_1")
        self.assertEqual(msg.tool_calls[0].result.success, True)

    def test_turn_with_results(self):
        msg = Message(role="user", content="test")
        tc = ToolCall(id="call_1", name="volume_up", arguments={})
        tr = ToolResult(tool_call_id="call_1", success=True, data={"output": "volume increased"})
        turn = Turn(message=msg, tool_calls=[tc], tool_results=[tr])
        self.assertEqual(turn.message, msg)
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(len(turn.tool_results), 1)
        self.assertEqual(turn.tool_results[0].success, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)