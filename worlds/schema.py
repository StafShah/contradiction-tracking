"""Data model for fictional document worlds with planted contradictions.

Follows CLAUDE.md section 4, extended with matched control facts, documents,
and a delivery schedule. Everything here is plain data so worlds serialize to
JSON and can be regenerated exactly from (seed, config).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Magnitude = Literal["qualifier", "numeric", "direct"]
Kind = Literal["resolve_by_rule", "resolve_by_credibility", "must_preserve"]
Handling = Literal["choose_a", "choose_b", "preserve_both"]
RelativeTo = Literal["before_interim", "after_interim", "before_compaction", "after_compaction"]
Credibility = Literal["high", "low", "equal"]
Modality = Literal["hedged", "certain"]

# Delivery windows for side B (see PILOT_PLAN.md section 3).
Window = Literal["before_interim", "after_interim_pre_compaction", "after_compaction"]


@dataclass
class Claim:
    text: str
    marker: str  # unique planted token: number, entity head token, or qualifier anchor
    source_id: str  # document id that carries this claim
    source_credibility: Credibility
    # Qualifier claims only: whether the anchor is stated with a hedge or as certain.
    modality: Modality | None = None
    # Qualifier hedged claims only: invented head token of the named condition
    # (e.g. "Zornan" in "only if the Zornan land purchase goes ahead"). It appears
    # only in the hedged document, so it is a deterministic marker for the hedge.
    condition_marker: str | None = None


@dataclass
class PlantedContradiction:
    id: str
    issue: str  # human-readable subject, e.g. "projected 2028 storage level of the Varn Reservoir"
    question_id: str
    side_a: Claim
    side_b: Claim
    magnitude: Magnitude
    kind: Kind
    correct_handling: Handling
    side_a_step: int
    inject_step: int  # step at which side B is delivered
    window: Window
    relative_to: RelativeTo
    recoverable: bool
    control_id: str  # matched control fact


@dataclass
class ControlFact:
    """Matched non-contradictory fact: two consistent mentions at the same steps
    as its matched contradiction's side A and side B."""

    id: str
    issue: str
    question_id: str
    first: Claim
    second: Claim
    magnitude: Magnitude
    first_step: int
    second_step: int
    recoverable: bool
    matched_contradiction_id: str


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    date: str  # ISO date inside the fictional world
    credibility: Credibility
    body: str
    recoverable: bool  # filed in the library (re-readable) or transient
    deliver_step: int | None  # None: in the library from the start
    role: Literal["side_a", "side_b", "control_first", "control_second", "distractor", "background"]
    item_id: str | None = None  # contradiction or control id this document carries

    def render(self) -> str:
        filing = "Filed in library" if self.recoverable else "Transient item: not filed in the library"
        return (
            f"[{self.doc_id}] {self.title}\n"
            f"Source: {self.source} | Date: {self.date} | {filing}\n"
            f"{self.body}"
        )


@dataclass
class Question:
    id: str
    text: str


@dataclass
class WorldConfig:
    n_steps: int = 40
    interim_step: int = 10
    compaction_steps: tuple[int, ...] = (15, 28)
    final_step: int = 40
    n_contradictions: int = 20
    recoverable: bool = False
    side_a_max_step: int = 6
    windows: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "before_interim": (3, 9),
            "after_interim_pre_compaction": (11, 14),
            "after_compaction": (16, 26),
        }
    )
    n_distractors_inbox: int = 14
    distractor_last_step: int = 34
    n_background_docs: int = 18
    max_world_tokens: int = 45_000


@dataclass
class World:
    seed: int
    config: WorldConfig
    authority: str
    brief: str
    questions: list[Question]
    documents: list[Document]
    contradictions: list[PlantedContradiction]
    controls: list[ControlFact]

    # -- convenience accessors -------------------------------------------------
    def doc(self, doc_id: str) -> Document:
        for d in self.documents:
            if d.doc_id == doc_id:
                return d
        raise KeyError(doc_id)

    def deliveries(self, step: int) -> list[Document]:
        return [d for d in self.documents if d.deliver_step == step]

    def library(self, step: int) -> list[Document]:
        """Documents re-readable at `step`: background docs plus delivered recoverable items."""
        return [
            d
            for d in self.documents
            if d.recoverable and (d.deliver_step is None or d.deliver_step <= step)
        ]

    def to_dict(self) -> dict:
        return asdict(self)
