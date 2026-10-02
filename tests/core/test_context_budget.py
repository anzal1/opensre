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


class TestThinkingBlocksAfterContextEdits:
    """Thinking blocks after a trimmed or truncated message no longer verify (#6497).

    Claude Sonnet 5.5, Opus 5.5 and Fable 5.1 reject such a request with a 400
    on accounts created on or after 2026-08-31, so the budget pass drops them.
    """

    @staticmethod
    def _transcript() -> list[dict]:
        def assistant(tool_id: str, thought: str) -> dict:
            return {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": thought, "signature": f"sig-{tool_id}"},
                    {"type": "tool_use", "id": tool_id, "name": "noop", "input": {}},
                ],
            }

        def result(tool_id: str, size: int) -> dict:
            return {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "x" * size}],
            }

        return [
            {"role": "user", "content": "investigate checkout-api latency"},
            assistant("t1", "look at metrics"),
            result("t1", 8000),
            assistant("t2", "look at logs"),
            result("t2", 8000),
            assistant("t3", "look at deploys"),
            result("t3", 100),
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "rolled back"},
                ],
            },
        ]

    @staticmethod
    def _has_thinking(message: dict) -> bool:
        content = message.get("content")
        return isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking")
            for block in content
        )

    def test_untouched_history_keeps_every_thinking_block(self) -> None:
        messages = self._transcript()
        original = json.loads(json.dumps(messages))

        enforce_context_budget(messages, fixed_overhead_tokens=0, ceiling=1_000_000)

        assert messages == original

    def test_only_blocks_with_an_intact_prefix_survive_a_trim(self) -> None:
        messages = self._transcript()
        original = json.loads(json.dumps(messages))

        enforce_context_budget(messages, fixed_overhead_tokens=0, ceiling=2_500)

        assert len(messages) < len(original)
        for idx, message in enumerate(messages):
            if self._has_thinking(message):
                # Every kept block must sit on exactly the prefix that produced it.
                assert messages[: idx + 1] == original[: idx + 1]
        # The edit is after the opening prompt, so the final turn always loses its block.
        assert not self._has_thinking(messages[-1])
        assert messages[-1]["content"] == [{"type": "text", "text": "rolled back"}]

    def test_truncation_also_drops_later_thinking_blocks(self) -> None:
        messages = self._transcript()[:3]
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "done", "signature": "sig-end"},
                    {"type": "text", "text": "summary"},
                ],
            }
        )

        # Only the opening prompt and one exchange: nothing to trim, so the
        # tool result is truncated in place instead.
        with patch("core.context_budget._tool_exchange_candidates", return_value=[]):
            enforce_context_budget(messages, fixed_overhead_tokens=0, ceiling=600)

        assert messages[2]["content"][0]["content"].endswith(_TRUNCATION_MARKER)
        assert self._has_thinking(messages[1])  # before the truncated message
        assert not self._has_thinking(messages[3])  # after it

    def test_shared_message_dicts_are_not_mutated(self) -> None:
        from core.context_budget import _strip_thinking_blocks_from

        transcript = self._transcript()
        request_copy = list(transcript)  # shallow, like the agent's converted copy

        _strip_thinking_blocks_from(request_copy, 0)

        assert all(not self._has_thinking(m) for m in request_copy)
        assert self._has_thinking(transcript[1])  # the stored transcript keeps its blocks

    def test_sdk_objects_and_bedrock_reasoning_blocks_are_recognised(self) -> None:
        from types import SimpleNamespace

        from core.context_budget import _strip_thinking_blocks_from

        tool_use = SimpleNamespace(type="tool_use", id="t1", name="noop", input={})
        messages = [
            {
                "role": "assistant",
                "content": [SimpleNamespace(type="thinking", thinking="x"), tool_use],
            },
            {
                "role": "assistant",
                "content": [{"reasoningContent": {"reasoningText": {"text": "x"}}}, {"text": "ok"}],
            },
        ]

        _strip_thinking_blocks_from(messages, 0)

        assert messages[0]["content"] == [tool_use]
        assert messages[1]["content"] == [{"text": "ok"}]

    def test_a_turn_that_is_only_thinking_is_left_alone(self) -> None:
        from core.context_budget import _strip_thinking_blocks_from

        only_thinking = {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]}
        messages = [only_thinking]

        _strip_thinking_blocks_from(messages, 0)

        assert messages == [only_thinking]
