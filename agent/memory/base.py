"""Memory policy interface.

A policy owns the agent's working context. The agent loop hands it every new
message (tagged with authorship), asks it for the context to send to the model,
and tells it when a compaction step has been reached. Only the memory policy
varies across conditions; the model, prompts, tools, and seeds stay fixed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)

from probes.passive import Segment, SegmentKind
from worlds.schema import World

# A summarizer takes the messages to compress plus a token budget and returns text.
Summarizer = Callable[[list[ChatMessage], int], Awaitable[str]]


@dataclass
class Entry:
    message: ChatMessage
    kind: SegmentKind  # authorship for the passive probe
    step: int


def message_text(m: ChatMessage) -> str:
    """Plain text of a message, including tool call arguments for assistant messages."""
    text = m.text if isinstance(m.content, list) else (m.content or "")
    if isinstance(m, ChatMessageAssistant) and m.tool_calls:
        calls = []
        for tc in m.tool_calls:
            args = "\n".join(f"{k}: {v}" for k, v in (tc.arguments or {}).items())
            calls.append(f"[tool call {tc.function}]\n{args}")
        text = (text + "\n\n" if text else "") + "\n\n".join(calls)
    return text


@dataclass
class CompactionRecord:
    step: int
    n_messages_in: int
    input_text_chars: int
    summary: str


class MemoryPolicy(ABC):
    name: str = "base"

    def __init__(self, world: World, system_prompt: str, summarizer: Summarizer | None = None,
                 summary_budget_tokens: int = 1000, keep_last_steps: int = 2):
        self.world = world
        self.system = ChatMessageSystem(content=system_prompt)
        self.summarizer = summarizer
        self.summary_budget_tokens = summary_budget_tokens
        self.keep_last_steps = keep_last_steps
        self.entries: list[Entry] = []
        self.summary: str | None = None
        self.compactions: list[CompactionRecord] = []
        self.current_step = 0

    # ------------------------------------------------------------------ input
    def add(self, message: ChatMessage, kind: SegmentKind, step: int) -> None:
        self.entries.append(Entry(message, kind, step))

    def on_step(self, step: int) -> None:
        """Called at the start of each step, before context() is read."""
        self.current_step = step

    # ----------------------------------------------------------------- output
    def pinned(self) -> str | None:
        return None

    def summary_message(self) -> ChatMessage | None:
        if self.summary is None:
            return None
        return ChatMessageUser(content=f"[Summary of your earlier work, written at compaction]\n{self.summary}")

    def pinned_message(self) -> ChatMessage | None:
        p = self.pinned()
        return ChatMessageUser(content=p) if p else None

    def context(self) -> list[ChatMessage]:
        msgs: list[ChatMessage] = [self.system]
        if (s := self.summary_message()) is not None:
            msgs.append(s)
        if (p := self.pinned_message()) is not None:
            msgs.append(p)
        msgs += [e.message for e in self.entries]
        return msgs

    def segments(self) -> list[Segment]:
        """The same context as context(), tagged by authorship for the passive probe."""
        segs = [Segment("brief", self.system.text)]
        if self.summary is not None:
            segs.append(Segment("summary", self.summary))
        if (p := self.pinned()) is not None:
            segs.append(Segment("pinned", p))
        segs += [Segment(e.kind, message_text(e.message), e.step) for e in self.entries]
        return segs

    # ------------------------------------------------------------- compaction
    @abstractmethod
    async def compact(self, step: int) -> None: ...


class FullContext(MemoryPolicy):
    name = "full_context"

    async def compact(self, step: int) -> None:
        return None


class LLMSummary(MemoryPolicy):
    """Summarize everything older than the last `keep_last_steps` steps."""

    name = "llm_summary"

    async def compact(self, step: int) -> None:
        if self.summarizer is None:
            raise RuntimeError("llm_summary needs a summarizer")
        cutoff = step - self.keep_last_steps  # entries with step <= cutoff are compressed
        old = [e for e in self.entries if e.step <= cutoff]
        keep = [e for e in self.entries if e.step > cutoff]
        if not old:
            return
        to_summarize: list[ChatMessage] = []
        if self.summary is not None:
            to_summarize.append(ChatMessageUser(content=f"[Previous summary]\n{self.summary}"))
        to_summarize += [e.message for e in old]
        summary = await self.summarizer(to_summarize, self.summary_budget_tokens)
        self.compactions.append(
            CompactionRecord(step, len(to_summarize), sum(len(message_text(m)) for m in to_summarize), summary)
        )
        self.summary = summary
        self.entries = keep


class OraclePin(LLMSummary):
    """llm_summary plus a pinned block, excluded from compaction, that quotes both
    sides of every ground-truth contradiction whose side B has arrived. It does not
    say which side is correct. Ceiling for perfect detection plus retention."""

    name = "oracle_pin"

    def pinned(self) -> str | None:
        active = [c for c in self.world.contradictions if c.inject_step <= self.current_step]
        if not active:
            return None
        lines = ["[Pinned record: claims on the same issue from different sources. This block is kept verbatim.]"]
        for c in sorted(active, key=lambda c: c.inject_step):
            da, db = self.world.doc(c.side_a.source_id), self.world.doc(c.side_b.source_id)
            lines.append(f"\nIssue: {c.issue}")
            lines.append(f"- {da.doc_id} ({da.source}, {da.date}): {c.side_a.text}")
            lines.append(f"- {db.doc_id} ({db.source}, {db.date}): {c.side_b.text}")
        return "\n".join(lines)


POLICIES: dict[str, type[MemoryPolicy]] = {
    "full_context": FullContext,
    "llm_summary": LLMSummary,
    "oracle_pin": OraclePin,
}
