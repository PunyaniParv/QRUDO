"""
SARV voice module.

Milestone 3 scope: mic -> wake word -> record until silence -> speech-to-text -> text.
No LLM, no tool execution here yet. This module's only job is to reliably turn
spoken audio into text and hand it off (via a callback) to whatever consumes it next.
"""