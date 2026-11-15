"""Final-answer scoring: deterministic, from planted markers in the memo.

Outcomes per contradiction:
    correct            handled as ground truth requires
    wrong_choice       resolve item, the agent kept the wrong side
    wrongly_preserved  resolve item, both sides reported without a resolution
    wrongly_resolved   must_preserve item, only one side kept or one rejected
    omitted            neither side appears in the memo
Controls: correct (marker present) or omitted.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from probes.passive import _chunks, _span_rejected, chunk_evidence, marker_regex, side_spans
from worlds.schema import PlantedContradiction, World

SUPERSEDED_RX = re.compile(r"\b(superseded|rejected|outdated|low[- ]credibility|not credible)\b", re.I)
DISPUTED_RX = re.compile(r"\bdisputed\b", re.I)


@dataclass
class FinalItemScore:
    id: str
    item_type: str
    outcome: str
    present_a: bool
    present_b: bool | None = None
    rejected_a: bool | None = None
    rejected_b: bool | None = None
    flagged_disputed: bool | None = None


def _rows_with(memo: str, spans_fn) -> list[str]:
    return [ln for ln in memo.splitlines() if "|" in ln and spans_fn(ln)]


def score_contradiction(c: PlantedContradiction, memo: str) -> FinalItemScore:
    sa = lambda t: side_spans(t, c.side_a, c.magnitude, c.side_b)  # noqa: E731
    sb = lambda t: side_spans(t, c.side_b, c.magnitude, c.side_a)  # noqa: E731
    pa, pb = bool(sa(memo)), bool(sb(memo))

    # A status cell only speaks for a side when the row mentions that side alone.
    rows_a = [r for r in _rows_with(memo, sa) if not sb(r)]
    rows_b = [r for r in _rows_with(memo, sb) if not sa(r)]
    rej_a = any(_span_rejected(memo, sp) for sp in sa(memo)) or any(SUPERSEDED_RX.search(r) for r in rows_a)
    rej_b = any(_span_rejected(memo, sp) for sp in sb(memo)) or any(SUPERSEDED_RX.search(r) for r in rows_b)
    for ch in _chunks(memo):
        ev = chunk_evidence(ch, c)
        if ev.resolved_to == "a":
            rej_b = True
        elif ev.resolved_to == "b":
            rej_a = True
    flagged = any(DISPUTED_RX.search(r) for r in _rows_with(memo, sa) + _rows_with(memo, sb)) or any(
        chunk_evidence(ch, c).flagged for ch in _chunks(memo)
    )

    keeps_a = pa and not rej_a
    keeps_b = pb and not rej_b
    if not pa and not pb:
        outcome = "omitted"
    elif c.correct_handling == "preserve_both":
        outcome = "correct" if (keeps_a and keeps_b) else "wrongly_resolved"
    else:
        want_a = c.correct_handling == "choose_a"
        right, wrong = (keeps_a, keeps_b) if want_a else (keeps_b, keeps_a)
        if right and not wrong:
            outcome = "correct"
        elif right and wrong:
            outcome = "wrongly_preserved"
        elif wrong:
            outcome = "wrong_choice"
        else:
            outcome = "omitted"  # only rejected mentions remain: no usable position
    return FinalItemScore(c.id, "contradiction", outcome, pa, pb, rej_a, rej_b, flagged)


def score_memo(world: World, memo: str | None) -> list[FinalItemScore]:
    memo = memo or ""
    out = [score_contradiction(c, memo) for c in world.contradictions]
    for k in world.controls:
        present = bool(marker_regex(k.first.marker).search(memo))
        out.append(FinalItemScore(k.id, "control", "correct" if present else "omitted", present))
    return out


def summarize(scores: list[FinalItemScore]) -> dict:
    c = [s for s in scores if s.item_type == "contradiction"]
    k = [s for s in scores if s.item_type == "control"]
    from collections import Counter

    return {
        "contradiction_accuracy": sum(s.outcome == "correct" for s in c) / max(1, len(c)),
        "contradiction_outcomes": dict(Counter(s.outcome for s in c)),
        "control_recall": sum(s.outcome == "correct" for s in k) / max(1, len(k)),
    }


def to_dicts(scores: list[FinalItemScore]) -> list[dict]:
    return [asdict(s) for s in scores]
