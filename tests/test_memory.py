"""Memory policy tests with a fake summarizer (no GPU, no model)."""

import asyncio

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall

from agent.memory.base import FullContext, LLMSummary, OraclePin, message_text
from worlds.generator import generate_world


def fake_summarizer(log):
    async def fn(messages, budget):
        log.append((len(messages), budget))
        return f"SUMMARY({len(messages)} msgs)"
    return fn


def fill(policy, steps):
    for t in steps:
        policy.on_step(t)
        policy.add(ChatMessageUser(content=f"step {t} inbox"), "env", t)
        policy.add(ChatMessageAssistant(content="", tool_calls=[ToolCall(id=f"c{t}", function="write_note", arguments={"text": f"note {t}"})]), "agent", t)
        policy.add(ChatMessageTool(content="Note saved.", tool_call_id=f"c{t}", function="write_note"), "env", t)


@pytest.fixture
def world():
    return generate_world(0)


def test_full_context_never_compacts(world):
    p = FullContext(world, "SYS")
    fill(p, range(1, 16))
    asyncio.run(p.compact(15))
    assert len(p.entries) == 45 and p.summary is None


def test_llm_summary_keeps_last_steps_and_system(world):
    log = []
    p = LLMSummary(world, "SYS", summarizer=fake_summarizer(log), keep_last_steps=2)
    fill(p, range(1, 16))
    asyncio.run(p.compact(15))
    assert {e.step for e in p.entries} == {14, 15}
    ctx = p.context()
    assert ctx[0].role == "system" and ctx[0].text == "SYS"
    assert "SUMMARY(39 msgs)" in ctx[1].text
    assert log == [(39, 1000)]


def test_second_compaction_includes_previous_summary(world):
    log = []
    p = LLMSummary(world, "SYS", summarizer=fake_summarizer(log), keep_last_steps=2)
    fill(p, range(1, 16))
    asyncio.run(p.compact(15))
    fill(p, range(16, 29))
    asyncio.run(p.compact(28))
    # previous summary (1) + steps 14..26 kept-then-compressed (13 steps x 3 msgs)
    assert log[1][0] == 1 + 13 * 3
    assert {e.step for e in p.entries} == {27, 28}
    assert len(p.compactions) == 2


def test_context_message_order_valid_after_compaction(world):
    p = LLMSummary(world, "SYS", summarizer=fake_summarizer([]), keep_last_steps=2)
    fill(p, range(1, 16))
    asyncio.run(p.compact(15))
    roles = [m.role for m in p.context()]
    # every tool message follows an assistant message
    for i, r in enumerate(roles):
        if r == "tool":
            assert roles[i - 1] == "assistant"


def test_oracle_pin_survives_compaction_and_only_after_injection(world):
    p = OraclePin(world, "SYS", summarizer=fake_summarizer([]))
    first = min(c.inject_step for c in world.contradictions)
    p.on_step(first - 1)
    assert p.pinned() is None
    fill(p, range(1, 16))
    asyncio.run(p.compact(15))
    pinned = p.pinned()
    active = [c for c in world.contradictions if c.inject_step <= 15]
    for c in active:
        assert c.side_a.text in pinned and c.side_b.text in pinned
    for c in world.contradictions:
        if c.inject_step > 15:
            assert c.side_b.text not in pinned
    assert any(s.kind == "pinned" for s in p.segments())
    # pinned block never states which side is correct
    assert "correct" not in pinned.lower() and "supersede" not in pinned.lower()


def test_segments_mirror_context(world):
    p = LLMSummary(world, "SYS", summarizer=fake_summarizer([]))
    fill(p, range(1, 4))
    segs = p.segments()
    assert segs[0].kind == "brief"
    assert [s.kind for s in segs[1:]] == ["env", "agent", "env"] * 3
    assert "note 2" in segs[5].text  # tool call arguments are visible to the passive probe


def test_message_text_includes_tool_args():
    m = ChatMessageAssistant(content="thinking", tool_calls=[ToolCall(id="x", function="write_note", arguments={"text": "41.7%"})])
    assert "41.7%" in message_text(m) and "thinking" in message_text(m)
