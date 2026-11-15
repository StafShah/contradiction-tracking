"""Agent tools and the document environment.

The agent loop executes tools itself (not through Inspect's tool runner) so that
inbox delivery, recoverability, and step accounting stay fully deterministic.
Tool schemas are passed to the model as Inspect ToolInfo objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from inspect_ai.tool import ToolInfo, ToolParam, ToolParams

from worlds.schema import Document, World


def _tool(name: str, description: str, params: dict[str, str]) -> ToolInfo:
    return ToolInfo(
        name=name,
        description=description,
        parameters=ToolParams(
            properties={k: ToolParam(type="string", description=v) for k, v in params.items()},
            required=list(params),
        ),
    )


TOOLS: list[ToolInfo] = [
    _tool("read_doc", "Read a document from the planning library by its id (for example DOC-123). Transient inbox items are not filed and cannot be read again.", {"doc_id": "Document id"}),
    _tool("search", "Keyword search over the planning library. Returns matching document ids, titles, and snippets.", {"query": "Search terms"}),
    _tool("write_note", "Write a note to your working notes. Notes stay in your context like any other message.", {"text": "Note text"}),
    _tool("conclude_interim", "Record your interim conclusion on each of the memo questions.", {"text": "Interim conclusion"}),
    _tool("final_answer", "Submit the final briefing memo, including the evidence table.", {"memo": "The full memo"}),
]
TOOL_NAMES = {t.name for t in TOOLS}


@dataclass
class ToolResult:
    text: str
    is_error: bool = False
    done: bool = False


@dataclass
class DocEnvironment:
    world: World
    notes: list[tuple[int, str]] = field(default_factory=list)
    interim: list[tuple[int, str]] = field(default_factory=list)
    final_memo: str | None = None
    reads: list[tuple[int, str, bool]] = field(default_factory=list)  # (step, doc_id, ok)
    searches: list[tuple[int, str, int]] = field(default_factory=list)  # (step, query, n_hits)
    early_final_attempts: list[int] = field(default_factory=list)

    def inbox(self, step: int) -> list[Document]:
        return self.world.deliveries(step)

    def execute(self, step: int, name: str, args: dict) -> ToolResult:
        if name not in TOOL_NAMES:
            return ToolResult(f"Unknown tool: {name}", is_error=True)
        try:
            return getattr(self, f"_t_{name}")(step, **{k: str(v) for k, v in (args or {}).items()})
        except TypeError as e:
            return ToolResult(f"Bad arguments for {name}: {e}", is_error=True)

    # ------------------------------------------------------------------ tools
    def _t_read_doc(self, step: int, doc_id: str) -> ToolResult:
        doc_id = doc_id.strip().upper()
        if not doc_id.startswith("DOC-"):
            doc_id = "DOC-" + doc_id.lstrip("DOC").lstrip("-")
        lib = {d.doc_id: d for d in self.world.library(step)}
        ok = doc_id in lib
        self.reads.append((step, doc_id, ok))
        if not ok:
            return ToolResult(f"{doc_id} is not in the library. It may be a transient item that was not filed.", is_error=True)
        return ToolResult(lib[doc_id].render())

    def _t_search(self, step: int, query: str) -> ToolResult:
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        scored = []
        for d in self.world.library(step):
            hay = (d.title + " " + d.body).lower()
            score = sum(hay.count(t) for t in terms)
            if score:
                scored.append((score, d))
        scored.sort(key=lambda x: (-x[0], x[1].doc_id))
        self.searches.append((step, query, len(scored)))
        if not scored:
            return ToolResult("No library documents matched.")
        lines = []
        for _, d in scored[:5]:
            snippet = d.body[:160].replace("\n", " ")
            lines.append(f"{d.doc_id} | {d.title} | {snippet}...")
        return ToolResult("\n".join(lines))

    def _t_write_note(self, step: int, text: str) -> ToolResult:
        self.notes.append((step, text))
        return ToolResult(f"Note saved ({len(self.notes)} notes so far).")

    def _t_conclude_interim(self, step: int, text: str) -> ToolResult:
        self.interim.append((step, text))
        return ToolResult("Interim conclusion recorded. Continue your analysis.")

    def _t_final_answer(self, step: int, memo: str) -> ToolResult:
        # Fixed horizon: survival analysis needs every run to last the same number of steps.
        final = self.world.config.final_step
        if step < final:
            self.early_final_attempts.append(step)
            return ToolResult(f"The memo is due at step {final}. Continue your analysis until then.", is_error=True)
        self.final_memo = memo
        return ToolResult("Memo submitted.", done=True)
