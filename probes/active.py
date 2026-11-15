"""Active probe: ask the agent about an issue in a discarded fork of its context.

The fork is the exact context the agent would see at that step, plus one
question. Its answer is recorded and never written back to the main
trajectory, so probing cannot contaminate the run (CLAUDE.md section 8).

Labeling is marker-first and deterministic, reusing the passive probe's side
detection on the answer text. Answers with no marker at all are flagged
`needs_judge` for later LLM-judge labeling (validated against hand labels).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass

from inspect_ai.model import ChatMessage, ChatMessageUser, GenerateConfig, Model

from probes.passive import PassiveProbe, chunk_evidence, marker_regex, side_spans
from worlds.schema import ControlFact, PlantedContradiction, World

PROBE_TEMPLATE = (
    "[Check-in question, answer from what you currently know. Do not call tools.]\n"
    "What is your current understanding of the {issue}, and which sources is it based on? "
    "Answer in two or three sentences."
)

DONT_KNOW_RX = re.compile(
    r"\b(don't know|do not know|not sure|no information|haven't seen|have not seen|unknown|not aware|"
    r"do not have|don't have|have no|no specific|not encountered|haven't encountered|no data|no record|"
    r"not found|cannot find|could not find|no documents?|none of the documents|not mentioned|not available)\b",
    re.I,
)


@dataclass
class ProbeAnswer:
    item_id: str
    item_type: str  # contradiction | control
    step: int
    question: str
    answer: str
    mentions_a: bool | None = None
    mentions_b: bool | None = None
    linked: bool | None = None
    flagged: bool | None = None
    resolved_to: str | None = None
    label: str | None = None  # same label space as the passive probe, or control present/absent
    says_dont_know: bool = False
    needs_judge: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None


def label_contradiction_answer(c: PlantedContradiction, answer: str) -> dict:
    pa = bool(side_spans(answer, c.side_a, c.magnitude, c.side_b))
    pb = bool(side_spans(answer, c.side_b, c.magnitude, c.side_a))
    ev = chunk_evidence(answer, c)
    label = PassiveProbe._label(c, pa, pb, ev.linked, ev.resolved_to)
    dont_know = bool(DONT_KNOW_RX.search(answer))
    return dict(mentions_a=pa, mentions_b=pb, linked=ev.linked, flagged=ev.flagged, resolved_to=ev.resolved_to,
                label=label, says_dont_know=dont_know, needs_judge=(not pa and not pb and not dont_know))


def label_control_answer(k: ControlFact, answer: str) -> dict:
    present = bool(marker_regex(k.first.marker).search(answer))
    dont_know = bool(DONT_KNOW_RX.search(answer))
    return dict(mentions_a=present, label="present" if present else "absent", says_dont_know=dont_know,
                needs_judge=(not present and not dont_know))


async def run_active_probe(
    model: Model,
    context: list[ChatMessage],
    world: World,
    step: int,
    tools: list,
    config: GenerateConfig,
    concurrency: int = 16,
) -> list[ProbeAnswer]:
    items: list[tuple[str, PlantedContradiction | ControlFact]] = []
    # The probe runs after step `step`'s inbox is in context and before the agent
    # acts, so items delivered at this step are included.
    for c in world.contradictions:
        if c.inject_step <= step:
            items.append(("contradiction", c))
    for k in world.controls:
        if k.second_step <= step:
            items.append(("control", k))

    sem = asyncio.Semaphore(concurrency)

    async def ask(kind: str, item) -> ProbeAnswer:
        q = PROBE_TEMPLATE.format(issue=item.issue)
        fork = list(context) + [ChatMessageUser(content=q)]
        async with sem:
            out = await model.generate(input=fork, tools=tools, tool_choice="none", config=config)
        text = out.completion or ""
        ans = ProbeAnswer(item.id, kind, step, q, text)
        if out.usage:
            ans.input_tokens, ans.output_tokens = out.usage.input_tokens, out.usage.output_tokens
        fields = label_contradiction_answer(item, text) if kind == "contradiction" else label_control_answer(item, text)
        for k_, v in fields.items():
            setattr(ans, k_, v)
        return ans

    return list(await asyncio.gather(*(ask(kind, it) for kind, it in items)))


def to_dicts(answers: list[ProbeAnswer]) -> list[dict]:
    return [asdict(a) for a in answers]
