"""Tests for the Phase C2 HTTP LLM provider (ai/provider_http.py).

These tests exercise the OpenAI-compatible HTTP provider using **fake HTTP
responses / transport injection** -- no internet, no API key, and no real
LLM are required.  They cover:

  - availability (disabled, missing config, configured)
  - no network when AI is disabled
  - normal text responses
  - tool-call parsing (single and multiple)
  - malformed responses (non-JSON, missing/invalid structure, bad arguments)
  - unknown tools (parsed by the provider, rejected by ToolRegistry)
  - provider errors (HTTP 401/403 auth, 500, network failure)
  - timeouts (bounded, explicit)
  - missing credentials (no endpoint/model, no api key)
  - API-key handling (Bearer header present, never leaked in output)
  - max-turn behaviour through the Assistant orchestrator
"""

from __future__ import annotations

import json
import socket
import unittest
import urllib.error
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai import AIConfig
from ai.assistant import Assistant
from ai.provider_http import DEFAULT_TIMEOUT, HttpProvider
from ai.schema import Message, ToolCall
from ai.tools.registry import (
    ensure_prebuilt_tools,
    get_registry,
    reset_registry,
)


class FakeResponse:
    """A stand-in for an HTTP response with status + read()."""

    def __init__(self, status=200, body=""):
        self.status = status
        self._body = body

    def read(self):
        return self._body


def openai_response(content=None, tool_calls=None, status=200):
    """Build a minimal, well-formed OpenAI chat-completions body."""
    msg = {"role": "assistant"}
    if content is not None:
        msg["content"] = content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return FakeResponse(
        status=status,
        body=json.dumps({"choices": [{"message": msg}]}),
    )


def tool_call(cid, name, arguments):
    """Build an OpenAI-style tool-call dict."""
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def make_transport(items):
    """Build (transport, captured, calls).

    ``items`` is a list where each element is either a ``FakeResponse`` to
    return or an ``Exception`` to raise, consumed in order (last repeats).
    Returns a transport suitable for ``HttpProvider(transport=...)``, a
    ``captured`` dict describing the request, and a ``calls`` counter.
    """
    captured = {}
    calls = {"count": 0}

    def transport(request, timeout):
        calls["count"] += 1
        captured["headers"] = dict(request.header_items())
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        body = request.data
        captured["body"] = body.decode("utf-8") if isinstance(body, bytes) else (body or "")
        item = items[min(len(items) - 1, calls["count"] - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    return transport, captured, calls


def cfg(**overrides):
    """A ready-to-use enabled AIConfig, overridable per test."""
    base = dict(
        ai_enabled=True,
        endpoint="https://api.example.com/v1/chat/completions",
        model="test-model",
        api_key="sk-test-123",
        timeout=30,
    )
    base.update(overrides)
    return AIConfig(**base)


def manifest_from_registry():
    """The exact manifest exposed by ToolRegistry."""
    reg = reset_and_ensure()
    return [
        {
            "name": name,
            "description": tool.description,
            "parameters": tool.parameters,
            "confirmation_required": tool.confirmation_required,
        }
        for name, tool in sorted(reg._tools.items())
    ]


def reset_and_ensure():
    reset_registry()
    return ensure_prebuilt_tools(get_registry())


class TestAvailability(unittest.TestCase):
    def test_not_available_when_disabled(self):
        p = HttpProvider(config=cfg(ai_enabled=False))
        self.assertFalse(p.available())

    def test_not_available_without_endpoint(self):
        p = HttpProvider(config=cfg(endpoint=""))
        self.assertFalse(p.available())

    def test_not_available_without_model(self):
        p = HttpProvider(config=cfg(model=""))
        self.assertFalse(p.available())

    def test_available_when_configured(self):
        p = HttpProvider(config=cfg())
        self.assertTrue(p.available())

    def test_available_without_api_key(self):
        # API key is optional (local OpenAI-compatible servers).
        p = HttpProvider(config=cfg(api_key=""))
        self.assertTrue(p.available())

    def test_available_never_raises(self):
        p = HttpProvider(config=None)
        # Not enabled by default; must be False, not an exception.
        self.assertFalse(p.available())


class TestNoNetworkWhenDisabled(unittest.TestCase):
    def test_respond_when_disabled_does_not_call_transport(self):
        transport, captured, calls = make_transport([openai_response(content="hi")])
        p = HttpProvider(config=cfg(ai_enabled=False), transport=transport)
        turn = p.respond([Message(role="user", content="hello")], [])
        self.assertEqual(calls["count"], 0)
        self.assertEqual(turn.message.content, "AI is currently unavailable")
        self.assertEqual(turn.tool_calls, [])

    def test_respond_unconfigured_does_not_call_transport(self):
        transport, captured, calls = make_transport([openai_response(content="hi")])
        p = HttpProvider(config=cfg(endpoint="", model=""), transport=transport)
        turn = p.respond([Message(role="user", content="hello")], [])
        self.assertEqual(calls["count"], 0)
        self.assertEqual(turn.message.content, "AI is currently unavailable")


class TestNormalTextResponse(unittest.TestCase):
    def _provider(self, items):
        transport, captured, calls = make_transport(items)
        p = HttpProvider(config=cfg(), transport=transport)
        return p, captured, calls

    def test_plain_text_parsed(self):
        p, captured, calls = self._provider([openai_response(content="Hello from QRUDO.")])
        turn = p.respond([Message(role="user", content="hi")], [])
        self.assertEqual(turn.tool_calls, [])
        self.assertEqual(turn.message.role, "assistant")
        self.assertEqual(turn.message.content, "Hello from QRUDO.")

    def test_request_shape(self):
        p, captured, calls = self._provider([openai_response(content="ok")])
        p.respond(
            [Message(role="system", content="be brief"),
             Message(role="user", content="turn it up")],
            manifest_from_registry(),
        )
        self.assertEqual(calls["count"], 1)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], cfg().endpoint)
        body = json.loads(captured["body"])
        self.assertEqual(body["model"], "test-model")
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["system", "user"])
        self.assertEqual(body["messages"][1]["content"], "turn it up")

    def test_tools_only_from_manifest(self):
        p, captured, calls = self._provider([openai_response(content="ok")])
        manifest = manifest_from_registry()
        p.respond([Message(role="user", content="x")], manifest)
        body = json.loads(captured["body"])
        tool_names = [t["function"]["name"] for t in body["tools"]]
        expected = sorted(t["name"] for t in manifest)
        self.assertEqual(tool_names, expected)

    def test_timeout_forwarded_and_bounded(self):
        p, captured, calls = self._provider([openai_response(content="ok")])
        p.respond([Message(role="user", content="x")], [])
        self.assertEqual(captured["timeout"], 30)

    def test_default_timeout_when_unset(self):
        transport, captured, calls = make_transport([openai_response(content="ok")])
        p = HttpProvider(config=cfg(timeout=0), transport=transport)
        self.assertEqual(p.timeout, DEFAULT_TIMEOUT)
        p.respond([Message(role="user", content="x")], [])
        self.assertEqual(captured["timeout"], DEFAULT_TIMEOUT)


class TestToolCallParsing(unittest.TestCase):
    def test_single_tool_call(self):
        transport, captured, calls = make_transport([
            openai_response(content="", tool_calls=[
                tool_call("call_1", "volume_up", {"step": 5}),
            ]),
        ])
        p = HttpProvider(config=cfg(), transport=transport)
        turn = p.respond([Message(role="user", content="louder")], [])
        self.assertEqual(len(turn.tool_calls), 1)
        tc = turn.tool_calls[0]
        self.assertIsInstance(tc, ToolCall)
        self.assertEqual(tc.id, "call_1")
        self.assertEqual(tc.name, "volume_up")
        self.assertEqual(tc.arguments, {"step": 5})

    def test_multiple_tool_calls(self):
        transport, captured, calls = make_transport([
            openai_response(content="", tool_calls=[
                tool_call("a", "volume_up", {"step": 1}),
                tool_call("b", "brightness_down", {"step": 2}),
                tool_call("c", "play_pause", {}),
            ]),
        ])
        p = HttpProvider(config=cfg(), transport=transport)
        turn = p.respond([Message(role="user", content="go")], [])
        self.assertEqual(len(turn.tool_calls), 3)
        self.assertEqual([t.name for t in turn.tool_calls],
                         ["volume_up", "brightness_down", "play_pause"])

    def test_nested_arguments_parsed(self):
        transport, captured, calls = make_transport([
            openai_response(content="", tool_calls=[
                tool_call("a", "catalog_action",
                          {"job_name": "Open Chrome", "app": "any"}),
            ]),
        ])
        p = HttpProvider(config=cfg(), transport=transport)
        turn = p.respond([Message(role="user", content="open chrome")], [])
        self.assertEqual(turn.tool_calls[0].arguments,
                         {"job_name": "Open Chrome", "app": "any"})

    def test_empty_arguments_tolerated(self):
        raw = {
            "id": "a", "type": "function",
            "function": {"name": "play_pause", "arguments": ""},
        }
        transport, captured, calls = make_transport([
            openai_response(content="", tool_calls=[raw]),
        ])
        p = HttpProvider(config=cfg(), transport=transport)
        turn = p.respond([Message(role="user", content="pause")], [])
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].arguments, {})



class TestMalformedResponses(unittest.TestCase):
    def _err_turn(self, item):
        transport, captured, calls = make_transport([item])
        p = HttpProvider(config=cfg(), transport=transport)
        return p.respond([Message(role="user", content="x")], [])

    def test_non_json_body(self):
        turn = self._err_turn(FakeResponse(200, "<html>not json"))
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("AI request failed", turn.message.content)

    def test_missing_choices(self):
        turn = self._err_turn(FakeResponse(200, json.dumps({"foo": 1})))
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("AI request failed", turn.message.content)

    def test_empty_choices(self):
        turn = self._err_turn(FakeResponse(200, json.dumps({"choices": []})))
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("AI request failed", turn.message.content)

    def test_message_not_object(self):
        body = json.dumps({"choices": [{"message": "oops"}]})
        turn = self._err_turn(FakeResponse(200, body))
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("AI request failed", turn.message.content)

    def test_tool_calls_not_list(self):
        body = json.dumps({"choices": [{"message": {
            "role": "assistant", "tool_calls": "nope"}}]})
        turn = self._err_turn(FakeResponse(200, body))
        self.assertEqual(turn.tool_calls, [])

    def test_invalid_arguments_json(self):
        raw = {"id": "a", "type": "function",
               "function": {"name": "volume_up", "arguments": "{nope"}}
        body = json.dumps({"choices": [{"message": {
            "role": "assistant", "tool_calls": [raw]}}]})
        turn = self._err_turn(FakeResponse(200, body))
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("AI request failed", turn.message.content)

    def test_arguments_not_object(self):
        raw = {"id": "a", "type": "function",
               "function": {"name": "volume_up", "arguments": "[1,2,3]"}}
        body = json.dumps({"choices": [{"message": {
            "role": "assistant", "tool_calls": [raw]}}]})
        turn = self._err_turn(FakeResponse(200, body))
        self.assertEqual(turn.tool_calls, [])

    def test_tool_call_missing_name(self):
        raw = {"id": "a", "type": "function",
               "function": {"arguments": "{}"}}
        body = json.dumps({"choices": [{"message": {
            "role": "assistant", "tool_calls": [raw]}}]})
        turn = self._err_turn(FakeResponse(200, body))
        self.assertEqual(turn.tool_calls, [])


class TestUnknownTools(unittest.TestCase):
    def test_provider_round_trips_unknown_name(self):
        transport, captured, calls = make_transport([
            openai_response(content="", tool_calls=[
                tool_call("a", "rm_rf_all", {"path": "/"}),
            ]),
        ])
        p = HttpProvider(config=cfg(), transport=transport)
        turn = p.respond([Message(role="user", content="delete")], [])
        self.assertEqual(turn.tool_calls[0].name, "rm_rf_all")

    def test_registry_rejects_unknown_tool(self):
        reg = reset_and_ensure()
        tc = ToolCall(id="a", name="rm_rf_all", arguments={"path": "/"})
        result = reg.call(None, tc)
        self.assertFalse(result.success)
        self.assertIn("not whitelisted", (result.error or "").lower())



class TestProviderErrors(unittest.TestCase):
    def _turn(self, item):
        transport, captured, calls = make_transport([item])
        p = HttpProvider(config=cfg(), transport=transport)
        return p.respond([Message(role="user", content="x")], [])

    def test_auth_failure_401(self):
        turn = self._turn(FakeResponse(401, json.dumps({"error": "bad key"})))
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("authentication failed", turn.message.content)
        self.assertIn("HTTP 401", turn.message.content)

    def test_auth_failure_403(self):
        turn = self._turn(FakeResponse(403, "forbidden"))
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("authentication failed", turn.message.content)

    def test_server_error_500(self):
        turn = self._turn(FakeResponse(500, "boom"))
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("HTTP 500", turn.message.content)

    def test_network_failure(self):
        turn = self._turn(urllib.error.URLError("connection refused"))
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("network or timeout", turn.message.content)

    def test_never_raises(self):
        # A response whose read() fails must still return a graceful Turn.
        class BadRead(FakeResponse):
            def read(self):
                raise OSError("io blew up")

        turn = self._turn(BadRead(200, "{}"))
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("AI request failed", turn.message.content)


class TestTimeout(unittest.TestCase):
    def test_socket_timeout_graceful(self):
        transport, captured, calls = make_transport([socket.timeout("timed out")])
        p = HttpProvider(config=cfg(), transport=transport)
        turn = p.respond([Message(role="user", content="x")], [])
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("network or timeout", turn.message.content)

    def test_timeout_error_graceful(self):
        transport, captured, calls = make_transport([TimeoutError("deadline")])
        p = HttpProvider(config=cfg(timeout=0.001), transport=transport)
        turn = p.respond([Message(role="user", content="x")], [])
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("network or timeout", turn.message.content)

    def test_non_http_scheme_refused(self):
        transport, captured, calls = make_transport([openai_response(content="hi")])
        p = HttpProvider(config=cfg(endpoint="file:///etc/passwd"), transport=transport)
        turn = p.respond([Message(role="user", content="x")], [])
        self.assertEqual(calls["count"], 0)  # transport never contacted
        self.assertEqual(turn.tool_calls, [])
        self.assertIn("AI request failed", turn.message.content)


class TestMissingCredentials(unittest.TestCase):
    def test_no_endpoint_graceful(self):
        transport, captured, calls = make_transport([openai_response(content="hi")])
        p = HttpProvider(config=cfg(endpoint="", model=""), transport=transport)
        turn = p.respond([Message(role="user", content="x")], [])
        self.assertEqual(calls["count"], 0)
        self.assertEqual(turn.message.content, "AI is currently unavailable")

    def test_no_api_key_does_not_crash(self):
        transport, captured, calls = make_transport([openai_response(content="ok")])
        p = HttpProvider(config=cfg(api_key=""), transport=transport)
        self.assertTrue(p.available())
        turn = p.respond([Message(role="user", content="x")], [])
        self.assertEqual(turn.message.content, "ok")
        # No Authorization header when no key is configured.
        self.assertNotIn("Authorization", captured["headers"])


class TestApiKeyHandling(unittest.TestCase):
    def test_bearer_header_present(self):
        transport, captured, calls = make_transport([openai_response(content="ok")])
        p = HttpProvider(config=cfg(api_key="sk-super-secret"), transport=transport)
        p.respond([Message(role="user", content="x")], [])
        self.assertEqual(captured["headers"]["Authorization"],
                         "Bearer sk-super-secret")

    def test_key_never_leaked_on_failure(self):
        # The server body echoes the key; it must be redacted from the Turn.
        leak = '{"error": "invalid key sk-super-secret"}'
        transport, captured, calls = make_transport([FakeResponse(401, leak)])
        p = HttpProvider(config=cfg(api_key="sk-super-secret"), transport=transport)
        turn = p.respond([Message(role="user", content="x")], [])
        self.assertNotIn("sk-super-secret", turn.message.content)
        self.assertIn("[REDACTED]", turn.message.content)

    def test_key_never_raised_in_exception_message(self):
        err = urllib.error.URLError("auth failed for sk-super-secret")
        transport, captured, calls = make_transport([err])
        p = HttpProvider(config=cfg(api_key="sk-super-secret"), transport=transport)
        turn = p.respond([Message(role="user", content="x")], [])
        self.assertNotIn("sk-super-secret", turn.message.content)


class TestMaxTurns(unittest.TestCase):
    def test_assistant_stops_after_max_turns(self):
        # Provider always emits a tool call to an unknown tool, so the
        # Assistant's tool-call loop continues until max_turns is reached.
        always_tool = openai_response(content="", tool_calls=[
            tool_call("a", "nonexistent_tool", {}),
        ])
        transport, captured, calls = make_transport([always_tool])
        config = cfg(max_turns=2)
        prov = HttpProvider(config=config, transport=transport)
        assistant = Assistant(config=config, provider=prov)
        result = assistant.escalate("please do the unrecognised thing")
        self.assertEqual(result,
                         "AI response exceeded maximum of 2 turns.")
        self.assertEqual(calls["count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

