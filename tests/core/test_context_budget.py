from __future__ import annotations

import json
from unittest.mock import patch

from core.context_budget import (
    _TOKENS_PER_CHAR,
    _TRUNCATION_MARKER,
    context_budget_ceiling_for_model,
    enforce_context_budget,
    estimate_message_tokens,
    strip_internal_message_markers,
    system_and_tools_overhead,
)


class TestGpt56ContextWindow:
    """GPT-5.6 ships a 1M-token window, unlike the 128k-pinned gpt-5 family (#3931)."""

    def test_sol_reclaims_the_million_token_window(self) -> None:
        # 1_000_000 window - 16_000 response headroom.
        assert context_budget_ceiling_for_model("gpt-5.6-sol") == 984_000

    def test_all_tiers_share_the_family_window(self) -> None:
        sol = context_budget_ceiling_for_model("gpt-5.6-sol")
        assert context_budget_ceiling_for_model("gpt-5.6-terra") == sol
        assert context_budget_ceiling_for_model("gpt-5.6-luna") == sol
        assert context_budget_ceiling_for_model("gpt-5.6") == sol

    def test_gpt56_is_not_shadowed_by_the_gpt5_catch_all(self) -> None:
        # ``_MODEL_CONTEXT_WINDOWS`` is matched by substring in insertion
        # order with a break, so moving the ``gpt-5.6`` key below ``gpt-5``
        # would silently pin 5.6 back to 128k. Nothing else would fail.
        assert context_budget_ceiling_for_model("gpt-5.6-sol") > context_budget_ceiling_for_model(
            "gpt-5.5"
        )

    def test_older_gpt5_models_keep_their_conservative_pin(self) -> None:
        # 128_000 window - 16_000 response headroom.
        assert context_budget_ceiling_for_model("gpt-5.5") == 112_000
        assert context_budget_ceiling_for_model("gpt-5") == 112_000


class TestGpt54ContextWindow:
    """GPT-5.4 is the hosted OpenAI default and must not inherit the 128k gpt-5 pin."""

    def test_default_mini_reclaims_the_million_token_window(self) -> None:
        assert context_budget_ceiling_for_model("gpt-5.4-mini") == 984_000
        assert context_budget_ceiling_for_model("gpt-5.4") == 984_000

    def test_gpt54_is_not_shadowed_by_the_gpt5_catch_all(self) -> None:
        assert context_budget_ceiling_for_model("gpt-5.4-mini") > context_budget_ceiling_for_model(
            "gpt-5.5"
        )


class TestGeminiContextWindow:
    """Gemini 1.5+ ships a 1M-token window; unknown models used to pin 128k."""

    def test_hosted_and_vertex_gemini_reclaim_the_million_token_window(self) -> None:
        # 1_000_000 window - 16_000 response headroom.
        ceiling = context_budget_ceiling_for_model("gemini-3.1-pro-preview")
        assert ceiling == 984_000
        assert context_budget_ceiling_for_model("gemini-2.5-pro") == ceiling
        assert context_budget_ceiling_for_model("gemini-3-flash-preview") == ceiling


def test_strip_internal_message_markers_removes_opensre_keys() -> None:
    messages = [
        {"role": "user", "content": "alert"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "seed", "name": "n", "input": {}}],
            "_opensre_seed": True,
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "seed", "content": "ok"}],
            "_opensre_seed": True,
            "_opensre_duplicate_result": True,
        },
    ]

    cleaned = strip_internal_message_markers(messages)

    assert cleaned[0] == messages[0]
    assert cleaned[1] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "seed", "name": "n", "input": {}}],
    }
    assert cleaned[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "seed", "content": "ok"}],
    }
    assert messages[1]["_opensre_seed"] is True
    assert messages[2]["_opensre_duplicate_result"] is True


def test_strip_internal_message_markers_preserves_other_underscore_keys() -> None:
    messages = [{"role": "user", "content": "hi", "_custom": "keep"}]

    cleaned = strip_internal_message_markers(messages)

    assert cleaned == [{"role": "user", "content": "hi", "_custom": "keep"}]


def test_estimate_message_tokens_includes_system_and_tools_overhead() -> None:
    messages = [{"role": "user", "content": "abcd"}]
    system = "sys"
    tools = [
        {"type": "function", "name": "alpha", "parameters": {"type": "object"}},
        {
            "type": "function",
            "name": "beta",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    ]

    message_tokens = int(len("abcd") * _TOKENS_PER_CHAR)
    system_tokens = int(len(system) * _TOKENS_PER_CHAR)
    tool_tokens = sum(
        int(len(json.dumps(schema, default=str)) * _TOKENS_PER_CHAR) for schema in tools
    )
    expected = message_tokens + system_tokens + tool_tokens

    assert estimate_message_tokens(messages, system=system, tools=tools) == expected


def test_estimate_message_tokens_distinguishes_distinct_tool_lists() -> None:
    messages: list[dict[str, str]] = []
    system = "system prompt"
    tools_a = [{"type": "function", "name": "alpha"}]
    tools_b = [{"type": "function", "name": "beta", "parameters": {"type": "object"}}]

    assert estimate_message_tokens(
        messages, system=system, tools=tools_a
    ) != estimate_message_tokens(messages, system=system, tools=tools_b)
    assert estimate_message_tokens(messages, system=system, tools=[]) == int(
        len(system) * _TOKENS_PER_CHAR
    )


def _tool_schema_dump_count(value: object) -> int:
    if isinstance(value, dict) and value.get("type") == "function":
        return 1
    if isinstance(value, list):
        return sum(1 for item in value if isinstance(item, dict) and item.get("type") == "function")
    return 0


def test_enforce_context_budget_serializes_tool_schemas_once_per_invocation() -> None:
    tools = [
        {"type": "function", "name": f"tool_{index}", "parameters": {"type": "object"}}
        for index in range(12)
    ]
    messages = [
        {"role": "user", "content": "alert"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "noop", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 4000}],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t2", "name": "noop", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "y" * 4000}],
        },
    ]
    ceiling = 500
    schema_dump_calls = 0
    original_dumps = json.dumps

    def counting_dumps(value: object, *args: object, **kwargs: object) -> str:
        nonlocal schema_dump_calls
        schema_dump_calls += _tool_schema_dump_count(value)
        return original_dumps(value, *args, **kwargs)

    with patch("core.context_budget.json.dumps", side_effect=counting_dumps):
        enforce_context_budget(messages, tools=tools, ceiling=ceiling)

    assert schema_dump_calls <= len(tools)


def test_enforce_context_budget_accepts_precomputed_overhead() -> None:
    tools = [
        {"type": "function", "name": f"tool_{index}", "parameters": {"type": "object"}}
        for index in range(12)
    ]
    messages = [
        {"role": "user", "content": "alert"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "noop", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 4000}],
        },
    ]
    ceiling = 500
    precomputed = system_and_tools_overhead(system="sys", tools=tools)
    schema_dump_calls = 0
    original_dumps = json.dumps

    def counting_dumps(value: object, *args: object, **kwargs: object) -> str:
        nonlocal schema_dump_calls
        schema_dump_calls += _tool_schema_dump_count(value)
        return original_dumps(value, *args, **kwargs)

    with patch("core.context_budget.json.dumps", side_effect=counting_dumps):
        enforce_context_budget(messages, fixed_overhead_tokens=precomputed, ceiling=ceiling)

    assert schema_dump_calls == 0


def test_enforce_context_budget_still_trims_when_over_ceiling_with_tools() -> None:
    tools = [{"type": "function", "name": "noop", "parameters": {"type": "object"}}]
    messages = [
        {"role": "user", "content": "alert"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "noop", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 4000}],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t2", "name": "noop", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "y" * 4000}],
        },
    ]
    ceiling = 500

    enforce_context_budget(messages, tools=tools, ceiling=ceiling)

    assert estimate_message_tokens(messages, tools=tools) <= ceiling
    assert len(messages) < 5


def _assistant(tool_id: str, thought: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": thought, "signature": f"sig-{tool_id}"},
            {"type": "tool_use", "id": tool_id, "name": "noop", "input": {}},
        ],
    }


def _result(tool_id: str, size: int) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "x" * size}],
    }


def _has_thinking(message: dict) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking")
        for block in content
    )


def _request_copy(transcript: list[dict]) -> list[dict]:
    """Mirror the agent loop: a fresh shallow copy of each message per request."""
    return [dict(message) for message in transcript]


class TestThinkingBlocksAfterContextEdits:
    """Thinking blocks after a trimmed or truncated message no longer verify (#6497).

    Claude Sonnet 5.5, Opus 5.5 and Fable 5.1 reject such a request with a 400
    on accounts created on or after 2026-08-31, so the budget pass drops them.
    """

    @staticmethod
    def _transcript() -> list[dict]:
        return [
            {"role": "user", "content": "investigate checkout-api latency"},
            _assistant("t1", "look at metrics"),
            _result("t1", 200),
            _assistant("t2", "look at logs"),
            _result("t2", 8000),  # largest exchange: evicted first
            _assistant("t3", "look at deploys"),
            _result("t3", 200),
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "rolled back"},
                ],
            },
        ]

    def test_untouched_history_keeps_every_thinking_block(self) -> None:
        messages = self._transcript()
        original = json.loads(json.dumps(messages))

        enforce_context_budget(messages, fixed_overhead_tokens=0, ceiling=1_000_000)

        assert messages == original

    def test_trim_keeps_blocks_before_the_edit_and_drops_the_rest(self) -> None:
        messages = self._transcript()
        original = json.loads(json.dumps(messages))

        enforce_context_budget(messages, fixed_overhead_tokens=0, ceiling=1_000)

        # Only the t2 exchange (indices 3-4) went.
        tool_ids = [
            block["id"]
            for m in messages
            if m["role"] == "assistant"
            for block in m["content"]
            if block.get("type") == "tool_use"
        ]
        assert tool_ids == ["t1", "t3"]
        assert messages[:3] == original[:3]  # untouched prefix, t1's block still there
        assert _has_thinking(messages[1])
        assert not _has_thinking(messages[3])  # t3 turn, produced after the edit point
        assert messages[3]["content"] == [original[5]["content"][1]]
        assert messages[-1]["content"] == [{"type": "text", "text": "rolled back"}]

    def test_truncation_drops_later_blocks_and_never_shrinks_thinking(self) -> None:
        messages = [
            {"role": "user", "content": "go"},
            _assistant("t1", "y" * 12000),  # largest message: tried first
            _result("t1", 8000),
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "done", "signature": "sig-end"},
                    {"type": "text", "text": "summary"},
                ],
            },
        ]

        with patch("core.context_budget._tool_exchange_candidates", return_value=[]):
            enforce_context_budget(messages, fixed_overhead_tokens=0, ceiling=2_500)

        assert messages[2]["content"][0]["content"].endswith(_TRUNCATION_MARKER)
        assert messages[1]["content"][0]["thinking"] == "y" * 12000  # signed text intact
        assert _has_thinking(messages[1])  # before the truncated message
        assert not _has_thinking(messages[3])  # after it

    def test_a_later_request_without_edits_does_not_resend_invalid_blocks(self) -> None:
        """Truncation persists in the shared transcript, so the strip must too."""
        transcript = [
            {"role": "user", "content": "go"},
            _assistant("t1", "first"),
            _result("t1", 8000),
            _assistant("t2", "second"),
            _result("t2", 100),
        ]

        with patch("core.context_budget._tool_exchange_candidates", return_value=[]):
            enforce_context_budget(
                _request_copy(transcript), fixed_overhead_tokens=0, ceiling=1_500
            )
            # Next request: the shrunken result now fits, so nothing is edited.
            second = _request_copy(transcript)
            enforce_context_budget(second, fixed_overhead_tokens=0, ceiling=1_500)

        assert transcript[2]["content"][0]["content"].endswith(_TRUNCATION_MARKER)
        assert _has_thinking(second[1])
        assert not _has_thinking(second[3])

    def test_sdk_objects_bedrock_and_litellm_spellings_are_recognised(self) -> None:
        from types import SimpleNamespace

        from core.context_budget import _strip_thinking_blocks_from

        tool_use = SimpleNamespace(type="tool_use", id="t1", name="noop", input={})
        litellm_blocks = [{"type": "thinking", "thinking": "x", "signature": "s"}]
        messages = [
            {
                "role": "assistant",
                "content": [SimpleNamespace(type="thinking", thinking="x"), tool_use],
            },
            {
                "role": "assistant",
                "content": [{"reasoningContent": {"reasoningText": {"text": "x"}}}, {"text": "ok"}],
            },
            {"role": "assistant", "content": "ok", "thinking_blocks": litellm_blocks},
        ]

        _strip_thinking_blocks_from(messages, 0)

        assert messages[0]["content"] == [tool_use]
        assert messages[1]["content"] == [{"text": "ok"}]
        assert litellm_blocks == []

    def test_a_turn_that_is_only_thinking_keeps_a_placeholder(self) -> None:
        from core.context_budget import _strip_thinking_blocks_from

        messages = [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]},
            {"role": "assistant", "content": [{"reasoningContent": {"reasoningText": {}}}]},
        ]

        _strip_thinking_blocks_from(messages, 0)

        assert messages[0]["content"] == [{"type": "text", "text": "[reasoning omitted]"}]
        assert messages[1]["content"] == [{"text": "[reasoning omitted]"}]

    def test_truncation_never_shrinks_bedrock_reasoning_text(self) -> None:
        reasoning = {"reasoningText": {"text": "y" * 12000, "signature": "sig"}}
        messages = [
            {"role": "user", "content": [{"text": "go"}]},
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": reasoning},
                    {"toolUse": {"toolUseId": "t1", "name": "noop", "input": {}}},
                ],
            },
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "x" * 8000}]}}],
            },
        ]

        with patch("core.context_budget._tool_exchange_candidates", return_value=[]):
            enforce_context_budget(messages, fixed_overhead_tokens=0, ceiling=4_000)

        assert reasoning["reasoningText"]["text"] == "y" * 12000
        assert messages[2]["content"][0]["toolResult"]["content"][0]["text"].endswith(
            _TRUNCATION_MARKER
        )
