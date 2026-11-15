"""Deterministic, template-based world generator.

generate_world(seed, config) -> World

No LLM is involved, so ground truth is exact. Same seed and config always
produce an identical world. See PILOT_PLAN.md section 3 for the design.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from worlds.schema import (
    Claim,
    ControlFact,
    Document,
    PlantedContradiction,
    Question,
    World,
    WorldConfig,
)
from worlds.templates import content as C

MAGNITUDES = ["qualifier", "numeric", "direct"]
KINDS = ["resolve_by_rule", "resolve_by_credibility", "must_preserve"]
WINDOW_ORDER = ["before_interim", "after_interim_pre_compaction", "after_compaction"]
RELATIVE_TO = {
    "before_interim": "before_interim",
    "after_interim_pre_compaction": "after_interim",
    "after_compaction": "after_compaction",
}
EXTRA_CELLS = [("direct", "must_preserve"), ("numeric", "resolve_by_rule")]

WORLD_YEAR_START = date(2027, 1, 4)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token for English prose)."""
    return max(1, len(text) // 4)


def _static_corpus() -> str:
    parts: list[str] = []
    for name in dir(C):
        val = getattr(C, name)
        if isinstance(val, (list, tuple, dict)):
            parts.append(repr(val))
    return " ".join(parts).lower()


_CORPUS = _static_corpus()


class _Names:
    """Invented, world-unique words. Used for entities and for marker head tokens."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: set[str] = set()

    def word(self) -> str:
        for _ in range(10_000):
            w = (
                self.rng.choice(C.ONSETS)
                + self.rng.choice(C.VOWELS)
                + self.rng.choice(C.CODAS)
                + self.rng.choice(C.ENDINGS)
            ).capitalize()
            # Guard: no prefix overlap so substring matching on one name never hits another.
            if 5 <= len(w) <= 10 and w.lower() not in _CORPUS and not any(w.lower() in u.lower() or u.lower() in w.lower() for u in self.used):
                self.used.add(w)
                return w
        raise RuntimeError("name space exhausted")


class _Numbers:
    """World-unique one-decimal numbers. Decimal markers avoid collisions with
    integers that appear in distractors, step counts, and doc ids."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used_int_parts: set[int] = set()

    def _fresh(self, lo: float, hi: float) -> float:
        for _ in range(10_000):
            v = round(self.rng.uniform(lo, hi), 1)
            ip = int(v)
            if v * 10 % 10 == 0:  # force a nonzero decimal digit
                continue
            if ip in self.used_int_parts:
                continue
            self.used_int_parts.add(ip)
            return v
        raise RuntimeError("number space exhausted")

    def value(self, lo: int, hi: int) -> float:
        return self._fresh(lo, hi)

    def discrepant(self, base: float, lo: int, hi: int) -> float:
        """A small discrepancy: 5 to 15 percent away from base, distinct integer part."""
        for _ in range(10_000):
            rel = self.rng.uniform(0.05, 0.15) * self.rng.choice([-1, 1])
            v = round(base * (1 + rel), 1)
            if v * 10 % 10 == 0 or int(v) in self.used_int_parts or v <= 0:
                continue
            self.used_int_parts.add(int(v))
            return v
        raise RuntimeError("could not find a discrepant value")


def fmt_num(v: float) -> str:
    return f"{v:.1f}"


def fmt_val(v: float, unit: str) -> str:
    return f"{fmt_num(v)}%" if unit == "%" else f"{fmt_num(v)} {unit}"


def _allocate_cells(seed: int) -> list[tuple[str, str, str]]:
    """Return 20 (magnitude, kind, window) triples.

    9 cells x 2, each cell's pair in different windows, rotated by seed, plus
    2 extra cells in the first and last windows (swapped on odd seeds).
    """
    out = []
    cells = [(m, k) for m in MAGNITUDES for k in KINDS]
    for c, (m, k) in enumerate(cells):
        w1 = (c + seed) % 3
        w2 = (c + seed + 1) % 3
        out.append((m, k, WINDOW_ORDER[w1]))
        out.append((m, k, WINDOW_ORDER[w2]))
    extra_windows = ["before_interim", "after_compaction"]
    if seed % 2 == 1:
        extra_windows.reverse()
    for (m, k), w in zip(EXTRA_CELLS, extra_windows):
        out.append((m, k, w))
    return out


def _assign_steps(rng: random.Random, items: list[int], lo: int, hi: int) -> dict[int, int]:
    """Spread items over steps lo..hi as evenly as possible, order shuffled."""
    steps = list(range(lo, hi + 1))
    rng.shuffle(steps)
    if len(items) <= len(steps):
        chosen = sorted(steps[: len(items)])
    else:
        chosen = sorted((steps * ((len(items) // len(steps)) + 1))[: len(items)])
    order = items[:]
    rng.shuffle(order)
    return dict(zip(order, chosen))


class _Builder:
    def __init__(self, seed: int, config: WorldConfig):
        self.seed = seed
        self.cfg = config
        self.rng = random.Random(seed)
        self.names = _Names(self.rng)
        self.nums = _Numbers(self.rng)
        self.authority = self.rng.choice(C.AUTHORITIES)
        self.docs: list[Document] = []
        self._doc_ids = self.rng.sample(range(100, 1000), 400)
        self._frame_uses: dict[int, int] = {}

    # ---------------------------------------------------------------- helpers
    def new_doc_id(self) -> str:
        return f"DOC-{self._doc_ids.pop()}"

    def rand_date(self, lo_day: int = 0, hi_day: int = 300) -> date:
        return WORLD_YEAR_START + timedelta(days=self.rng.randint(lo_day, hi_day))

    def pick_frame(self, question_id: str, magnitude: str) -> int:
        idxs = [i for i, f in enumerate(C.FRAMES) if f.question_id == question_id and f.magnitude == magnitude]
        # least-used frame first, ties broken randomly, for variety
        self.rng.shuffle(idxs)
        idxs.sort(key=lambda i: self._frame_uses.get(i, 0))
        self._frame_uses[idxs[0]] = self._frame_uses.get(idxs[0], 0) + 1
        return idxs[0]

    def filler(self, n: int = 2) -> str:
        return " ".join(self.rng.sample(C.FILLER, n))

    # ------------------------------------------------------- claim realization
    def realize_values(self, frame: C.Frame, magnitude: str, contradiction: bool, a_modality: str | None = None):
        """Return (marker_a, val_text_a, marker_b, val_text_b, cond_text, cond_head).

        For controls (contradiction=False) side b equals side a.
        """
        if magnitude == "numeric":
            va = self.nums.value(frame.lo, frame.hi)
            vb = self.nums.discrepant(va, frame.lo, frame.hi) if contradiction else va
            return fmt_num(va), fmt_val(va, frame.unit), fmt_num(vb), fmt_val(vb, frame.unit), None, None
        if magnitude == "direct":
            ha = self.names.word()
            hb = self.names.word() if contradiction else ha
            return ha, frame.value_form.format(h=ha), hb, frame.value_form.format(h=hb), None, None
        # qualifier: one anchor value, modality differs between sides
        v = self.nums.value(frame.lo, frame.hi)
        head = self.names.word()
        cond = frame.condition.format(c=head)
        return fmt_num(v), fmt_val(v, frame.unit), fmt_num(v), fmt_val(v, frame.unit), cond, head

    def sentence(self, magnitude: str, src: str, subj: str, val: str, frame: C.Frame, modality: str | None, cond: str | None, second_control: bool = False) -> str:
        if magnitude == "numeric":
            tpl = self.rng.choice(C.CONTROL_SECOND_NUMERIC if second_control else C.NUMERIC_TEMPLATES)
            return tpl.format(src=src, subj=subj, val=val)
        if magnitude == "direct":
            tpl = self.rng.choice(C.CONTROL_SECOND_DIRECT if second_control else C.DIRECT_TEMPLATES)
            return tpl.format(src=src, subj=subj, val=val)
        tpl = self.rng.choice(C.QUALIFIER_HEDGED if modality == "hedged" else C.QUALIFIER_CERTAIN)
        return tpl.format(src=src, subj=subj, val=val, verb=frame.verb, cond=cond)

    def make_doc(self, *, title: str, source: str, d: date, cred: str, claim_sentence: str, step: int, role: str, item_id: str) -> Document:
        body = f"{claim_sentence} {self.filler(2)}"
        doc = Document(
            doc_id=self.new_doc_id(),
            title=title,
            source=source,
            date=d.isoformat(),
            credibility=cred,
            body=body,
            recoverable=self.cfg.recoverable,
            deliver_step=step,
            role=role,
            item_id=item_id,
        )
        self.docs.append(doc)
        return doc

    # ------------------------------------------------------------- top level
    def build(self) -> World:
        cfg = self.cfg
        questions = [Question(qid, text) for qid, text in C.QUESTIONS]

        cells = _allocate_cells(self.seed)
        assert len(cells) == cfg.n_contradictions, "pilot allocation assumes 20 contradictions"

        # Question assignment: cycle a shuffled question order so each gets 3 or 4.
        qids = [q.id for q in questions]
        q_cycle = []
        while len(q_cycle) < len(cells):
            block = qids[:]
            self.rng.shuffle(block)
            q_cycle.extend(block)
        q_cycle = q_cycle[: len(cells)]

        # Side B steps per window.
        b_step: dict[int, int] = {}
        for w in WINDOW_ORDER:
            idx = [i for i, c in enumerate(cells) if c[2] == w]
            lo, hi = cfg.windows[w]
            b_step.update(_assign_steps(self.rng, idx, lo, hi))

        contradictions: list[PlantedContradiction] = []
        controls: list[ControlFact] = []

        for i, (mag, kind, window) in enumerate(cells):
            cid, ctl_id = f"C{i:02d}", f"K{i:02d}"
            qid = q_cycle[i]
            sb = b_step[i]
            sa = self.rng.randint(1, min(cfg.side_a_max_step, sb - 1))

            # ---- contradiction
            frame = C.FRAMES[self.pick_frame(qid, mag)]
            subj = frame.subject.format(e=self.names.word())
            a_mod = b_mod = None
            if mag == "qualifier":
                a_mod = self.rng.choice(["hedged", "certain"])
                b_mod = "certain" if a_mod == "hedged" else "hedged"
            ma, va, mb, vb, cond, chead = self.realize_values(frame, mag, contradiction=True)

            handling, (src_a, src_b) = self.sources_for(kind, qid)
            sent_a = self.sentence(mag, src_a["phrase"], subj, va, frame, a_mod, cond)
            sent_b = self.sentence(mag, src_b["phrase"], subj, vb, frame, b_mod, cond)
            doc_a = self.make_doc(title=src_a["title"], source=src_a["source"], d=src_a["date"], cred=src_a["cred"], claim_sentence=sent_a, step=sa, role="side_a", item_id=cid)
            doc_b = self.make_doc(title=src_b["title"], source=src_b["source"], d=src_b["date"], cred=src_b["cred"], claim_sentence=sent_b, step=sb, role="side_b", item_id=cid)

            contradictions.append(
                PlantedContradiction(
                    id=cid,
                    issue=subj,
                    question_id=qid,
                    side_a=Claim(sent_a, ma, doc_a.doc_id, src_a["cred"], a_mod, chead if a_mod == "hedged" else None),
                    side_b=Claim(sent_b, mb, doc_b.doc_id, src_b["cred"], b_mod, chead if b_mod == "hedged" else None),
                    magnitude=mag,
                    kind=kind,
                    correct_handling=handling,
                    side_a_step=sa,
                    inject_step=sb,
                    window=window,
                    relative_to=RELATIVE_TO[window],
                    recoverable=cfg.recoverable,
                    control_id=ctl_id,
                )
            )

            # ---- matched control: same question, magnitude, steps; consistent mentions
            cframe = C.FRAMES[self.pick_frame(qid, mag)]
            csubj = cframe.subject.format(e=self.names.word())
            c_mod = self.rng.choice(["hedged", "certain"]) if mag == "qualifier" else None
            cm, cv, _, _, ccond, cchead = self.realize_values(cframe, mag, contradiction=False)
            s1, s2 = self.rng.sample(C.CONTROL_SOURCES, 2)
            d1 = self.rand_date(0, 200)
            d2 = d1 + timedelta(days=self.rng.randint(10, 60))
            csent1 = self.sentence(mag, f"{self.authority} {s1}", csubj, cv, cframe, c_mod, ccond)
            csent2 = self.sentence(mag, f"{self.authority} {s2}", csubj, cv, cframe, c_mod, ccond, second_control=True)
            cdoc1 = self.make_doc(title=f"{s1} update", source=f"{self.authority} {s1}", d=d1, cred="high", claim_sentence=csent1, step=sa, role="control_first", item_id=ctl_id)
            cdoc2 = self.make_doc(title=f"{s2} update", source=f"{self.authority} {s2}", d=d2, cred="high", claim_sentence=csent2, step=sb, role="control_second", item_id=ctl_id)
            controls.append(
                ControlFact(
                    id=ctl_id,
                    issue=csubj,
                    question_id=qid,
                    first=Claim(csent1, cm, cdoc1.doc_id, "high", c_mod, cchead if c_mod == "hedged" else None),
                    second=Claim(csent2, cm, cdoc2.doc_id, "high", c_mod, cchead if c_mod == "hedged" else None),
                    magnitude=mag,
                    first_step=sa,
                    second_step=sb,
                    recoverable=cfg.recoverable,
                    matched_contradiction_id=cid,
                )
            )

        self.add_distractors()
        self.add_background()
        brief = self.make_brief(questions)
        world = World(
            seed=self.seed,
            config=cfg,
            authority=self.authority,
            brief=brief,
            questions=questions,
            documents=sorted(self.docs, key=lambda d: (d.deliver_step or 0, d.doc_id)),
            contradictions=contradictions,
            controls=controls,
        )
        return world

    def sources_for(self, kind: str, qid: str):
        """Return (correct_handling, (source_spec_a, source_spec_b))."""
        if kind == "resolve_by_rule":
            series = C.SERIES[qid]
            ed_old = self.rng.randint(2, 6)
            d_old = self.rand_date(0, 150)
            d_new = d_old + timedelta(days=self.rng.randint(60, 120))
            old = {"source": f"{self.authority} {series}", "title": f"{series}, Edition {ed_old}", "date": d_old, "cred": "equal",
                   "phrase": f"{series} (Edition {ed_old})"}
            new = {"source": f"{self.authority} {series}", "title": f"{series}, Edition {ed_old + 1}", "date": d_new, "cred": "equal",
                   "phrase": f"{series} (Edition {ed_old + 1})"}
            # Which side is the newer edition is randomized, so arrival order and
            # recency are decorrelated.
            if self.rng.random() < 0.5:
                return "choose_b", (old, new)
            return "choose_a", (new, old)
        if kind == "resolve_by_credibility":
            hi_src = self.rng.choice(C.HIGH_CRED_SOURCES)
            lo_src, lo_title = self.rng.choice(C.LOW_CRED_SOURCES)
            hi = {"source": hi_src, "title": f"{hi_src} note", "date": self.rand_date(), "cred": "high", "phrase": hi_src}
            lo = {"source": lo_src, "title": lo_title, "date": self.rand_date(), "cred": "low", "phrase": lo_src}
            if self.rng.random() < 0.5:
                return "choose_a", (hi, lo)
            return "choose_b", (lo, hi)
        # must_preserve: two independent, equally credible consultancies, close dates
        c1, c2 = self.rng.sample(C.CONSULTANCIES, 2)
        d = self.rand_date(30, 250)
        a = {"source": c1, "title": f"{c1} study", "date": d, "cred": "equal", "phrase": f"{c1} study commissioned by the Authority"}
        b = {"source": c2, "title": f"{c2} study", "date": d + timedelta(days=self.rng.randint(0, 14)), "cred": "equal",
             "phrase": f"independent {c2} study commissioned by the Authority"}
        return "preserve_both", (a, b)

    def add_distractors(self) -> None:
        picks = self.rng.sample(C.DISTRACTORS, min(self.cfg.n_distractors_inbox, len(C.DISTRACTORS)))
        for title, source, body in picks:
            step = self.rng.randint(1, self.cfg.distractor_last_step)
            self.docs.append(
                Document(
                    doc_id=self.new_doc_id(),
                    title=title,
                    source=f"{self.authority} {source}",
                    date=self.rand_date().isoformat(),
                    credibility="equal",
                    body=body,
                    recoverable=self.cfg.recoverable,
                    deliver_step=step,
                    role="distractor",
                )
            )

    def add_background(self) -> None:
        districts = [self.names.word() for _ in range(3)]
        reservoirs = [self.names.word() for _ in range(2)]
        plants = [self.names.word() for _ in range(2)]
        fill = dict(auth=self.authority, d1=districts[0], d2=districts[1], d3=districts[2], r1=reservoirs[0], r2=reservoirs[1], p1=plants[0], p2=plants[1])
        titles = self.rng.sample(C.BACKGROUND_TITLES, self.cfg.n_background_docs)
        for title in titles:
            paras = self.rng.sample(C.BACKGROUND_PARAGRAPHS, 4)
            body = "\n\n".join(p.format(**fill) for p in paras)
            self.docs.append(
                Document(
                    doc_id=self.new_doc_id(),
                    title=title,
                    source=f"{self.authority} Planning Library",
                    date=self.rand_date(0, 30).isoformat(),
                    credibility="high",
                    body=body,
                    recoverable=True,
                    deliver_step=None,
                    role="background",
                )
            )

    def make_brief(self, questions: list[Question]) -> str:
        cfg = self.cfg
        qtext = "\n".join(f"{q.id}. {q.text}" for q in questions)
        return f"""You are a policy analyst at the {self.authority}. The board has asked for a briefing memo answering these questions:

{qtext}

How this assignment works:
- You have {cfg.n_steps} steps. Each step is one tool call.
- New items arrive in your inbox over time and are shown to you automatically after your tool calls. Some items are transient (phone readouts, forwarded messages) and are not filed in the library; once they leave your context you cannot read them again. Use write_note to record anything you need.
- The planning library holds background documents. Use search and read_doc to consult it.
- At step {cfg.interim_step} you will be asked to record an interim conclusion with conclude_interim. At step {cfg.final_step} you must submit the memo with final_answer.

Evidence rules set by the board:
- When a later edition of the same report series revises a figure, the later edition supersedes the earlier one.
- Treat the Authority Audit Office, the Provincial Statistics Bureau, and the Office of the Chief Engineer as authoritative. Treat anonymous posts, campaign leaflets, lobby newsletters, and unsourced social media as low credibility.
- Where two credible, independent sources disagree and no rule settles it, do not pick one. Report the point as disputed and attribute both positions.

Memo format for final_answer:
1. A short answer to each question (Q1 to Q6), with reasoning.
2. An evidence table with one row per figure or claim you relied on, in the form:
   question | claim | sources (document ids) | status (settled, disputed, or superseded)
"""


def generate_world(seed: int, config: WorldConfig | None = None) -> World:
    return _Builder(seed, config or WorldConfig()).build()


def world_token_estimate(world: World) -> int:
    """Estimate of the tokens the agent would see if every document were in context."""
    return estimate_tokens(world.brief) + sum(estimate_tokens(d.render()) for d in world.documents)
