"""Passive probe: a deterministic scan of what the model saw at each step.

Input per step is the exact context, as a list of Segments tagged by who wrote
them. Output is one state label per contradiction (and a presence record per
control fact). No model calls.

Labels (CLAUDE.md section 4):
    undetected      both sides present, but no agent-authored text links them
    held_both       both sides present and linked in agent-authored text
    collapsed_to_a  only side A present, no recorded resolution toward A
    collapsed_to_b  only side B present, no recorded resolution toward B
    resolved_correct / resolved_wrong
                    the agent recorded a resolution (one side rejected), and
                    the chosen side is still present
    lost            neither side present
Steps before side B arrives get "pre_injection".

Known limitations (unit-tested and documented so they are not a surprise):
- Qualifier sides share one anchor value. The hedged side is detected by its
  condition token or by a hedge word in the anchor's clause; the certain side
  is an anchor occurrence whose clause has neither. Paraphrase that drops both
  the hedge and the condition reads as the certain side. That is intended:
  dropping a hedge is a collapse to the certain claim.
- Resolution detection uses local cue phrases ("superseded by", "outdated",
  "low credibility", a table status cell). Unusual phrasings are missed and
  read as held_both. The active probe is the check on this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from worlds.schema import Claim, PlantedContradiction, World
from worlds.templates.content import HEDGE_WORDS

SegmentKind = Literal["brief", "env", "agent", "summary", "pinned"]
AGENT_KINDS = {"agent", "summary", "pinned"}

Label = Literal[
    "pre_injection", "undetected", "held_both", "collapsed_to_a", "collapsed_to_b",
    "resolved_correct", "resolved_wrong", "lost",
]


@dataclass
class Segment:
    kind: SegmentKind
    text: str
    step: int | None = None


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

def marker_regex(marker: str) -> re.Pattern:
    """Numbers match exactly with digit boundaries; names match as whole words, any case."""
    if re.fullmatch(r"\d+\.\d", marker):
        return re.compile(r"(?<![\d.])" + re.escape(marker) + r"(?!\d)")
    return re.compile(r"\b" + re.escape(marker) + r"\b", re.IGNORECASE)


_HEDGES = [h.strip() for h in HEDGE_WORDS if h.strip() != "may"]
HEDGE_RX = re.compile(r"\b(" + "|".join(re.escape(h) for h in _HEDGES) + r")\b", re.IGNORECASE)
# "may" is a hedge only in lower case (the month is capitalized).
MAY_RX = re.compile(r"\bmay\b")
CLAUSE_SPLIT_RX = re.compile(r"(?<!\d)\.(?!\d)|[;\n|]")

REJECT_AFTER_RX = re.compile(
    r"^[^.;\n]{0,40}?\b(superseded|outdated|obsolete|incorrect|unreliable|not credible|"
    r"low[- ]credibility|discarded|disregard(?:ed)?|retracted|rejected|withdrawn|"
    r"old(?:er)? (?:figure|value|edition|estimate)|no longer valid|out of date|not reliable)\b",
    re.IGNORECASE,
)
REJECT_BEFORE_RX = re.compile(
    r"\b(old|older|earlier|previous|prior|outdated|superseded|obsolete|rejected|discarded|"
    r"unreliable|low[- ]credibility|not credible|disregard)\b[^.;\n]{0,25}$",
    re.IGNORECASE,
)
CHOSEN_BEFORE_RX = re.compile(
    r"(\b(superseded|replaced|overridden|corrected|revised|updated|amended)\s+(by|to|with)\b[^.;\n]{0,30}"
    r"|\b(in favou?r of|use|adopt|prefer|accept|go with|rely on)\b[^.;\n]{0,20}"
    # "the current reliable figure is 64.4%" (active probe, smoke test 2026-11-08)
    r"|\b(current|correct|valid|reliable|latest|authoritative|accepted|applicable|operative)\s+(?:\w+\s+){0,2}"
    r"(figure|value|estimate|rate|number|level|cost|position)\s+(?:is|of|stands at|=)\s*[^.;\n]{0,12})$",
    re.IGNORECASE,
)
DISPUTE_RX = re.compile(
    r"\b(conflict\w*|contradict\w*|discrepan\w*|disagree\w*|inconsistent|differ\w*|disputed|"
    r"unresolved|competing|vs\.?|versus|two estimates)\b",
    re.IGNORECASE,
)


def _clause_bounds(text: str, pos: int) -> tuple[int, int]:
    start = 0
    for m in CLAUSE_SPLIT_RX.finditer(text, 0, pos):
        start = m.end()
    m = CLAUSE_SPLIT_RX.search(text, pos)
    end = m.start() if m else len(text)
    return start, end


def _line_bounds(text: str, s: int, e: int) -> tuple[int, int]:
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", e)
    return ls, (le if le != -1 else len(text))


def _is_hedged_clause(clause: str) -> bool:
    return bool(HEDGE_RX.search(clause) or MAY_RX.search(clause))


# Hedge attachment. Within one sentence (or table cell), each hedge word is
# attached to one decimal number, so a hedge on a neighbouring list item does not
# leak onto this anchor (smoke test 2026-11-08: "A >116.7 ML/day (DOC-76), B likely
# >81.1 ML/day conditional on ..."). Leading hedges modify the next number;
# trailing conditions modify the previous one. Falls back to the other direction.
DECIMAL_RX = re.compile(r"(?<![\d.])\d+\.\d(?!\d)")
TRAILING_HEDGES = {"only if", "if", "provided", "provided that", "conditional", "condition on", "conditioned",
                   "contingent", "subject to", "depending", "depends", "assuming", "unless"}


def _hedge_matches(sentence: str):
    for m in HEDGE_RX.finditer(sentence):
        yield m, m.group(1).lower() in TRAILING_HEDGES
    for m in MAY_RX.finditer(sentence):
        yield m, False


def _anchor_hedged(text: str, pos: int) -> bool:
    ss, se = _clause_bounds(text, pos)
    sentence = text[ss:se]
    rel = pos - ss
    nums = [m.start() for m in DECIMAL_RX.finditer(sentence)]
    if rel not in nums:  # anchor not parsed as a decimal (should not happen); fall back to sentence-level
        return _is_hedged_clause(sentence)
    for hm, trailing in _hedge_matches(sentence):
        before = [n for n in nums if n < hm.start()]
        after = [n for n in nums if n >= hm.end()]
        if trailing:
            target = before[-1] if before else (after[0] if after else None)
        else:
            target = after[0] if after else (before[-1] if before else None)
        if target == rel:
            return True
    return False


# ---------------------------------------------------------------------------
# Side detection
# ---------------------------------------------------------------------------

def side_spans(text: str, claim: Claim, magnitude: str, other: Claim | None = None) -> list[tuple[int, int]]:
    """Spans in `text` that express `claim`.

    For qualifier claims, `other` is the opposite side (used to exclude its
    condition token from the certain side's clauses).
    """
    if magnitude != "qualifier":
        return [m.span() for m in marker_regex(claim.marker).finditer(text)]

    anchor_rx = marker_regex(claim.marker)
    cond = claim.condition_marker or (other.condition_marker if other else None)
    cond_rx = marker_regex(cond) if cond else None
    spans: list[tuple[int, int]] = []
    if claim.modality == "hedged":
        if cond_rx:
            spans += [m.span() for m in cond_rx.finditer(text)]
        for m in anchor_rx.finditer(text):
            if _anchor_hedged(text, m.start()):
                spans.append(m.span())
    else:
        for m in anchor_rx.finditer(text):
            if _anchor_hedged(text, m.start()):
                continue
            # the item's own condition anywhere in the sentence marks it hedged
            ss, se = _clause_bounds(text, m.start())
            if cond_rx and cond_rx.search(text[ss:se]):
                continue
            spans.append(m.span())
    return sorted(set(spans))


def _span_rejected(text: str, span: tuple[int, int]) -> bool:
    s, e = span
    ls, le = _line_bounds(text, s, e)
    before = text[max(ls, s - 50): s]
    after = text[e: min(le, e + 70)]
    if CHOSEN_BEFORE_RX.search(before):
        return False
    return bool(REJECT_AFTER_RX.search(after) or REJECT_BEFORE_RX.search(before))


def _span_chosen(text: str, span: tuple[int, int]) -> bool:
    s, _ = span
    ls, _ = _line_bounds(text, s, s)
    return bool(CHOSEN_BEFORE_RX.search(text[max(ls, s - 50): s]))


def _chunks(text: str) -> list[str]:
    """Paragraphs (blank-line separated)."""
    return [c for c in re.split(r"\n\s*\n", text) if c.strip()]


@dataclass
class ChunkEvidence:
    linked: bool = False
    flagged: bool = False  # dispute language in a linked chunk
    resolved_to: str | None = None  # "a" or "b" if this step's text records a resolution


def chunk_evidence(chunk: str, c: PlantedContradiction) -> ChunkEvidence:
    sa = side_spans(chunk, c.side_a, c.magnitude, c.side_b)
    sb = side_spans(chunk, c.side_b, c.magnitude, c.side_a)
    ev = ChunkEvidence()
    if not (sa and sb):
        return ev
    ev.linked = True
    ev.flagged = bool(DISPUTE_RX.search(chunk))
    rej_a = any(_span_rejected(chunk, sp) for sp in sa) or any(_span_chosen(chunk, sp) for sp in sb)
    rej_b = any(_span_rejected(chunk, sp) for sp in sb) or any(_span_chosen(chunk, sp) for sp in sa)
    if rej_a and not rej_b:
        ev.resolved_to = "b"
    elif rej_b and not rej_a:
        ev.resolved_to = "a"
    return ev


# ---------------------------------------------------------------------------
# Stateful labeler
# ---------------------------------------------------------------------------

@dataclass
class ContradictionStep:
    id: str
    step: int
    label: Label
    present_a: bool = False
    present_b: bool = False
    agent_a: bool = False
    agent_b: bool = False
    linked: bool = False
    flagged: bool = False
    resolution_seen: str | None = None
    resolved_to: str | None = None  # sticky state after this step


@dataclass
class ControlStep:
    id: str
    step: int
    active: bool  # second mention delivered
    present: bool
    agent_present: bool
    modality_preserved: bool | None = None  # qualifier only, agent text


@dataclass
class StepResult:
    step: int
    contradictions: dict[str, ContradictionStep] = field(default_factory=dict)
    controls: dict[str, ControlStep] = field(default_factory=dict)


class PassiveProbe:
    def __init__(self, world: World):
        self.world = world
        self._resolved_to: dict[str, str | None] = {c.id: None for c in world.contradictions}
        self.history: list[StepResult] = []

    def step(self, step: int, segments: list[Segment]) -> StepResult:
        res = StepResult(step=step)
        all_text = [s.text for s in segments]
        agent_text = [s.text for s in segments if s.kind in AGENT_KINDS]
        agent_chunks = [ch for t in agent_text for ch in _chunks(t)]

        for c in self.world.contradictions:
            if step < c.inject_step:
                res.contradictions[c.id] = ContradictionStep(c.id, step, "pre_injection")
                continue
            pa = any(side_spans(t, c.side_a, c.magnitude, c.side_b) for t in all_text)
            pb = any(side_spans(t, c.side_b, c.magnitude, c.side_a) for t in all_text)
            aa = any(side_spans(t, c.side_a, c.magnitude, c.side_b) for t in agent_text)
            ab = any(side_spans(t, c.side_b, c.magnitude, c.side_a) for t in agent_text)

            linked = flagged = False
            seen: set[str] = set()
            for ch in agent_chunks:
                ev = chunk_evidence(ch, c)
                linked |= ev.linked
                flagged |= ev.flagged
                if ev.resolved_to:
                    seen.add(ev.resolved_to)
            resolution_seen = seen.pop() if len(seen) == 1 else None

            state = self._resolved_to[c.id]
            if resolution_seen:
                state = resolution_seen
            elif linked and flagged and not seen and pa and pb:
                state = None  # conflict reopened as disputed
            self._resolved_to[c.id] = state

            label = self._label(c, pa, pb, linked, state)
            res.contradictions[c.id] = ContradictionStep(
                c.id, step, label, pa, pb, aa, ab, linked, flagged, resolution_seen, state
            )

        for k in self.world.controls:
            active = step >= k.second_step
            mod_ok = None
            if k.magnitude == "qualifier":
                # Presence is the anchor in any modality; modality is tracked separately.
                anchor = marker_regex(k.first.marker)
                present = any(anchor.search(t) for t in all_text)
                agent_present = any(anchor.search(t) for t in agent_text)
                if agent_present:
                    # Preserved iff agent text never states the anchor in the opposite modality
                    # (for a hedged control, a bare anchor counts as a dropped hedge).
                    opposite = Claim("", k.first.marker, "", "high", "certain" if k.first.modality == "hedged" else "hedged")
                    mod_ok = not any(side_spans(t, opposite, "qualifier", k.first) for t in agent_text)
            else:
                present = any(side_spans(t, k.first, k.magnitude) for t in all_text)
                agent_present = any(side_spans(t, k.first, k.magnitude) for t in agent_text)
            res.controls[k.id] = ControlStep(k.id, step, active, present, agent_present, mod_ok)

        self.history.append(res)
        return res

    @staticmethod
    def _label(c: PlantedContradiction, pa: bool, pb: bool, linked: bool, state: str | None) -> Label:
        def resolved(side: str) -> Label:
            correct = (side == "a" and c.correct_handling == "choose_a") or (side == "b" and c.correct_handling == "choose_b")
            return "resolved_correct" if correct else "resolved_wrong"

        if pa and pb:
            if state:
                return resolved(state)
            return "held_both" if linked else "undetected"
        if pa or pb:
            side = "a" if pa else "b"
            if state == side:
                return resolved(side)
            return "collapsed_to_a" if pa else "collapsed_to_b"
        return "lost"
