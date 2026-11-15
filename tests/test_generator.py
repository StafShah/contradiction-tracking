import json
import re
from collections import Counter

import pytest

from worlds.generator import MAGNITUDES, KINDS, generate_world, world_token_estimate
from worlds.schema import WorldConfig

SEEDS = [0, 1, 2, 3, 4]


@pytest.fixture(scope="module", params=SEEDS)
def world(request):
    return generate_world(request.param)


def test_deterministic():
    a = json.dumps(generate_world(7).to_dict(), sort_keys=True)
    b = json.dumps(generate_world(7).to_dict(), sort_keys=True)
    assert a == b


def test_different_seeds_differ():
    assert json.dumps(generate_world(1).to_dict()) != json.dumps(generate_world(2).to_dict())


def test_counts(world):
    assert len(world.contradictions) == 20
    assert len(world.controls) == 20
    assert len(world.questions) == 6


def test_all_cells_covered_twice(world):
    cells = Counter((c.magnitude, c.kind) for c in world.contradictions)
    for m in MAGNITUDES:
        for k in KINDS:
            assert cells[(m, k)] >= 2
    assert cells[("direct", "must_preserve")] == 3
    assert cells[("numeric", "resolve_by_rule")] == 3


def test_window_counts(world):
    w = Counter(c.window for c in world.contradictions)
    assert w == {"before_interim": 7, "after_interim_pre_compaction": 6, "after_compaction": 7}


def test_cell_pairs_in_different_windows(world):
    by_cell = {}
    for c in world.contradictions:
        by_cell.setdefault((c.magnitude, c.kind), []).append(c.window)
    for windows in by_cell.values():
        assert len(set(windows[:2])) == 2


def test_rotation_across_seeds():
    """Across the 5 pilot seeds, every cell appears in every window."""
    seen = {}
    for s in SEEDS:
        for c in generate_world(s).contradictions:
            seen.setdefault((c.magnitude, c.kind), set()).add(c.window)
    for windows in seen.values():
        assert len(windows) == 3


def test_steps_respect_windows_and_order(world):
    cfg = world.config
    for c in world.contradictions:
        lo, hi = cfg.windows[c.window]
        assert lo <= c.inject_step <= hi
        assert 1 <= c.side_a_step < c.inject_step
        assert c.side_a_step <= cfg.side_a_max_step
        assert c.inject_step != cfg.interim_step
        assert c.inject_step not in cfg.compaction_steps


def test_relative_to(world):
    for c in world.contradictions:
        if c.window == "before_interim":
            assert c.relative_to == "before_interim" and c.inject_step < world.config.interim_step
        else:
            assert c.inject_step > world.config.interim_step
        if c.window == "after_compaction":
            assert c.inject_step > world.config.compaction_steps[0]
        if c.window == "after_interim_pre_compaction":
            assert c.inject_step < world.config.compaction_steps[0]


def test_controls_matched(world):
    by_id = {c.id: c for c in world.contradictions}
    assert {k.matched_contradiction_id for k in world.controls} == set(by_id)
    for k in world.controls:
        c = by_id[k.matched_contradiction_id]
        assert c.control_id == k.id
        assert k.magnitude == c.magnitude
        assert k.question_id == c.question_id
        assert k.first_step == c.side_a_step
        assert k.second_step == c.inject_step
        assert k.first.marker == k.second.marker
        assert k.recoverable == c.recoverable
        if k.magnitude == "qualifier":
            assert k.first.modality == k.second.modality


def test_sides_differ_correctly(world):
    for c in world.contradictions:
        if c.magnitude == "qualifier":
            assert c.side_a.marker == c.side_b.marker
            assert {c.side_a.modality, c.side_b.modality} == {"hedged", "certain"}
        else:
            assert c.side_a.marker != c.side_b.marker


def test_numeric_discrepancy_is_small(world):
    for c in world.contradictions:
        if c.magnitude == "numeric":
            a, b = float(c.side_a.marker), float(c.side_b.marker)
            assert 0.03 <= abs(a - b) / a <= 0.17


def test_correct_handling_matches_kind(world):
    for c in world.contradictions:
        if c.kind == "must_preserve":
            assert c.correct_handling == "preserve_both"
            assert c.side_a.source_credibility == c.side_b.source_credibility == "equal"
        elif c.kind == "resolve_by_credibility":
            chosen = c.side_a if c.correct_handling == "choose_a" else c.side_b
            other = c.side_b if c.correct_handling == "choose_a" else c.side_a
            assert chosen.source_credibility == "high" and other.source_credibility == "low"
        else:
            da, db = world.doc(c.side_a.source_id), world.doc(c.side_b.source_id)
            newer = "choose_a" if da.date > db.date else "choose_b"
            assert c.correct_handling == newer
            assert da.source == db.source  # same series


def _marker_regex(marker: str) -> re.Pattern:
    if re.fullmatch(r"\d+\.\d", marker):
        return re.compile(r"(?<![\d.])" + re.escape(marker) + r"(?!\d)")
    return re.compile(r"\b" + re.escape(marker) + r"\b", re.IGNORECASE)


def test_markers_unique_and_located(world):
    """Each marker appears only in the documents that carry it."""
    items = []
    for c in world.contradictions:
        items.append((c.side_a.marker, {c.side_a.source_id, c.side_b.source_id} if c.magnitude == "qualifier" else {c.side_a.source_id}))
        if c.magnitude != "qualifier":
            items.append((c.side_b.marker, {c.side_b.source_id}))
    for k in world.controls:
        items.append((k.first.marker, {k.first.source_id, k.second.source_id}))
    markers = [m for m, _ in items]
    assert len(markers) == len(set(markers)), "duplicate markers in world"
    for marker, expected_docs in items:
        rx = _marker_regex(marker)
        hits = {d.doc_id for d in world.documents if rx.search(d.render())}
        assert hits == expected_docs, (marker, hits, expected_docs)


def test_no_marker_in_brief(world):
    for c in world.contradictions:
        for m in (c.side_a.marker, c.side_b.marker):
            assert not _marker_regex(m).search(world.brief)
        assert c.issue not in world.brief


def test_brief_has_questions_and_rules(world):
    for q in world.questions:
        assert q.text in world.brief
    assert "later edition" in world.brief
    assert "disputed" in world.brief


def test_pilot_all_non_recoverable(world):
    for d in world.documents:
        if d.deliver_step is not None:
            assert d.recoverable is False
    assert all(not c.recoverable for c in world.contradictions)


def test_recoverable_flag_propagates():
    w = generate_world(0, WorldConfig(recoverable=True))
    assert all(c.recoverable for c in w.contradictions)
    lib_ids = {d.doc_id for d in w.library(40)}
    for c in w.contradictions:
        assert c.side_a.source_id in lib_ids and c.side_b.source_id in lib_ids


def test_library_excludes_transient(world):
    lib = {d.doc_id for d in world.library(40)}
    for c in world.contradictions:
        assert c.side_a.source_id not in lib and c.side_b.source_id not in lib


def test_qualifier_text_modality(world):
    from worlds.templates.content import HEDGE_WORDS

    hedge_rx = re.compile(r"\b(" + "|".join(re.escape(h.strip()) for h in HEDGE_WORDS) + r")\b")
    for c in world.contradictions:
        if c.magnitude != "qualifier":
            continue
        for side in (c.side_a, c.side_b):
            has_hedge = bool(hedge_rx.search(side.text))
            assert has_hedge == (side.modality == "hedged"), side.text


def test_token_budget(world):
    assert world_token_estimate(world) <= world.config.max_world_tokens


def test_question_spread(world):
    q = Counter(c.question_id for c in world.contradictions)
    assert set(q) == {f"Q{i}" for i in range(1, 7)}
    assert max(q.values()) - min(q.values()) <= 1


def test_condition_marker_only_in_hedged_docs(world):
    for c in world.contradictions:
        if c.magnitude != "qualifier":
            assert c.side_a.condition_marker is None and c.side_b.condition_marker is None
            continue
        hedged = c.side_a if c.side_a.modality == "hedged" else c.side_b
        certain = c.side_b if hedged is c.side_a else c.side_a
        assert hedged.condition_marker and certain.condition_marker is None
        rx = _marker_regex(hedged.condition_marker)
        hits = {d.doc_id for d in world.documents if rx.search(d.render())}
        assert hits == {hedged.source_id}
