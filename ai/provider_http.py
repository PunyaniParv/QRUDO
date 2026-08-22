"""Provider-neutral HTTP LLM provider for QRUDO.

Implements the OpenAI-compatible Chat Completions / tool-calling HTTP
interface using **only the Python standard library** (``urllib.request``).
It is completely optional: importing this module and constructing the
provider performs no network I/O, and the provider never reaches the
network unless ``respond()`` is called on a provider that ``available()``
reports as ready.

Safety properties (see docs/QRUDO_INTELLIGENCE_V1.md §9, §12-16):

  - **No network when AI is disabled.** ``available()`` returns False and
    ``respond()`` returns a graceful no-op Turn without calling the
    transport.
  - **API keys come only from config/environment and are never logged** and
    never echoed back in errors or responses.
  - **Explicit, bounded timeouts** on every request.
  - **Graceful failure.** Network failure, missing/empty response, malformed
    JSON, invalid provider output, authentication failure, and timeout all
    return a ``Turn`` with an explanatory message and **no** tool calls.
    ``respond()`` never raises into the Assistant loop.
  - **Provider-neutral output.** ``respond()`` returns only the existing
    ``Turn`` / ``ToolCall`` structures from ``ai.schema``.
  - **Manifest-only tools.** The provider receives the tool manifest from
    ``ToolRegistry`` via its caller and serialises exactly that; it never
    invents tools and never bypasses ``ToolRegistry``.
  - **http/https only.** Unsupported URL schemes are refused.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List

from ai.config import AIConfig
from ai.provider import AssistantProvider
from ai.schema import Message, ToolCall, Turn

logger = logging.getLogger("qrudo.ai.provider_http")

#: Default bounded per-request timeout (seconds).
DEFAULT_TIMEOUT = 60

#: Provider-config keys forwarded into the request body if supplied.
_EXTRA_KEYS = ("temperature", "max_tokens", "top_p")

#: Only http/https endpoints are ever contacted.
_ALLOWED_SCHEMES = ("http", "https")


class ProviderError(Exception):
    """A provider request could not be completed.

    Always caught inside :meth:`HttpProvider.respond` so it never
    propagates into the Assistant loop.
    """


def _default_transport(request: Any, timeout: float) -> Any:
    """Standard-library HTTP transport (overrideable in tests).

    ``urlopen`` with an explicit bounded timeout raises ``URLError`` /
    ``socket.timeout`` (both ``OSError`` subclasses) on any network or
    timeout problem, which the provider converts into a graceful Turn.
    """
    return urllib.request.urlopen(request, timeout=timeout)


class HttpProvider(AssistantProvider):
    """OpenAI-compatible HTTP LLM provider (provider-neutral).

    Constructor arguments are all optional and default from ``AIConfig``:

      - ``config``:    an ``AIConfig`` carrying endpoint/model/api_key/timeout.
      - ``endpoint``:  override the chat-completions URL.
      - ``model``:     override the model name.
      - ``api_key``:   override the API key (never logged).
      - ``timeout``:   bounded per-request timeout in seconds.
      - ``transport``: ``callable(request, timeout) -> response-like`` with
        ``.status`` and ``.read()``.  Defaults to ``urllib.request.urlopen``
        so tests can inject fake HTTP responses with no network and no LLM.
    """

    def __init__(
        self,
        *,
        config: AIConfig | None = None,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config or AIConfig.load()
        self.endpoint = endpoint if endpoint is not None else self._config.endpoint
        self.model = model if model is not None else self._config.model
        self.api_key = api_key if api_key is not None else getattr(
            self._config, "api_key", "")
        try:
            self.timeout = float(
                timeout if timeout is not None
                else getattr(self._config, "timeout", DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            self.timeout = float(DEFAULT_TIMEOUT)
        if self.timeout <= 0:
            self.timeout = float(DEFAULT_TIMEOUT)
        self._transport = transport or _default_transport

    # ------------------------------------------------------------------
    # AssistantProvider protocol
    # ------------------------------------------------------------------

    def available(self) -> bool:
        """True when the provider is configured enough to attempt a request.

        Never performs a network call and never raises.  Requires AI to be
        enabled and an endpoint + model to be configured.  An API key is
        optional because some OpenAI-compatible local servers need none.
        """
        try:
            if not self._config.ai_enabled:
                return False
            if not self.endpoint or not self.model:
                return False
            return True
        except Exception:
            return False

    def respond(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        **config: Any,
    ) -> Turn:
        """Generate a ``Turn`` from the conversation and registry manifest.

        Fails gracefully: any network/parse/auth/timeout problem returns a
        ``Turn`` with an explanatory message and no tool calls, never
        raising and never contacting the network when AI is disabled.
        """
        if not self.available():
            return self._unavailable()

        try:
            payload = self._build_payload(messages, tools, config)
            request = self._build_request(payload)

            try:
                response = self._transport(request, self.timeout)
            except ProviderError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise ProviderError(f"network or timeout error: {exc}") from exc

            try:
                status = int(getattr(response, "status", 200))
            except (TypeError, ValueError):
                status = 200
            try:
                body = response.read()
            except Exception as exc:  # a transport that lied about read()
                raise ProviderError(
                    f"failed to read provider response: {exc}") from exc
            if isinstance(body, bytes):
                try:
                    body = body.decode("utf-8", "replace")
                except Exception:
                    body = ""

            if status < 200 or status >= 300:
                raise ProviderError(self._status_message(status, body))

            try:
                data = json.loads(body)
            except (ValueError, TypeError) as exc:
                raise ProviderError(
                    f"invalid JSON from provider: {exc}") from exc

            return self._parse_response(data)

        except ProviderError as exc:
            return self._error_turn(str(exc))
        except Exception as exc:  # absolute backstop -- never crash QRUDO
            logger.debug("http provider unexpected error: %s", type(exc).__name__)
            return self._error_turn("unexpected provider failure")



    # ------------------------------------------------------------------
    # Payload / request construction
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.model:
            payload["model"] = self.model

        msgs: List[Dict[str, Any]] = []
        for m in messages:
            entry: Dict[str, Any] = {"role": m.role}
            if m.content:
                entry["content"] = m.content
            if m.tool_calls:
                entry["tool_calls"] = [
                    self._openai_tool_call(tc) for tc in m.tool_calls
                ]
            msgs.append(entry)
        payload["messages"] = msgs

        if tools:
            payload["tools"] = [self._openai_tool(t) for t in tools]

        for key in _EXTRA_KEYS:
            if key in config and config[key] is not None:
                payload[key] = config[key]

        return payload

    def _openai_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        name = tool.get("name")
        if not name:
            raise ProviderError("tool manifest entry missing 'name'")
        function: Dict[str, Any] = {
            "name": str(name),
            "description": str(tool.get("description", "")),
        }
        if isinstance(tool.get("parameters"), dict):
            function["parameters"] = tool["parameters"]
        return {"type": "function", "function": function}

    def _openai_tool_call(self, tc: ToolCall) -> Dict[str, Any]:
        return {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments),
            },
        }

    def _build_request(self, payload: Dict[str, Any]) -> Any:
        parsed = urllib.parse.urlparse(self.endpoint or "")
        scheme = parsed.scheme.lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise ProviderError(
                f"unsupported endpoint scheme: {scheme or '(none)'}")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        data = json.dumps(payload).encode("utf-8")
        return urllib.request.Request(
            self.endpoint, data=data, headers=headers, method="POST")



    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, data: Any) -> Turn:
        if not isinstance(data, dict):
            raise ProviderError("provider response is not a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("provider response missing 'choices'")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            raise ProviderError("provider response has invalid 'message'")

        content = message.get("content")
        content = content if isinstance(content, str) else (content or "")

        tool_calls: List[ToolCall] = []
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ProviderError("provider response has invalid 'tool_calls'")
        for raw in raw_calls:
            if not isinstance(raw, dict):
                raise ProviderError("invalid tool call in provider response")
            call_id = str(raw.get("id") or f"call_{len(tool_calls)}")
            fn = raw.get("function") if isinstance(raw.get("function"), dict) else {}
            name = fn.get("name")
            if not name or not isinstance(name, str):
                raise ProviderError("tool call missing function name")
            args_str = fn.get("arguments")
            if isinstance(args_str, str) and args_str.strip():
                try:
                    args = json.loads(args_str)
                except (ValueError, TypeError) as exc:
                    raise ProviderError(
                        f"invalid tool call arguments for '{name}': {exc}") from exc
            else:
                args = {}
            if not isinstance(args, dict):
                raise ProviderError(
                    f"tool call arguments for '{name}' must be an object")
            tool_calls.append(
                ToolCall(id=call_id, name=name, arguments=args))

        msg = Message(role="assistant", content=str(content), tool_calls=tool_calls)
        return Turn(message=msg, tool_calls=tool_calls)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _status_message(self, status: int, body: Any) -> str:
        snippet = ""
        if isinstance(body, str) and body.strip():
            snippet = " " + body[:200]
        if status in (401, 403):
            return f"authentication failed (HTTP {status}){snippet}"
        if 400 <= status < 500:
            return f"provider rejected the request (HTTP {status}){snippet}"
        return f"provider error (HTTP {status}){snippet}"

    def _redact(self, text: str) -> str:
        if self.api_key:
            return text.replace(self.api_key, "[REDACTED]")
        return text

    def _error_turn(self, message: str) -> Turn:
        return Turn(
            message=Message(
                role="assistant",
                content=f"AI request failed: {self._redact(message)}",
            ),
            tool_calls=[],
        )

    def _unavailable(self) -> Turn:
        return Turn(
            message=Message(
                role="assistant", content="AI is currently unavailable"),
            tool_calls=[],
        )


def create_http_provider(config: AIConfig | None = None) -> HttpProvider:
    """Factory for an :class:`HttpProvider` from an ``AIConfig``.

    Convenience for wiring; the provider remains fully inactive (no network)
    unless AI is enabled and credentials/endpoint are configured.
    """
    return HttpProvider(config=config or AIConfig.load())

