"""Handwritten fixtures for every passive-probe label, plus documented limitations."""

import pytest

from probes.passive import PassiveProbe, Segment, marker_regex, side_spans
from worlds.schema import Claim, ControlFact, PlantedContradiction, World, WorldConfig

INJECT = 5


def _c(cid, mag, kind, handling, a: Claim, b: Claim) -> PlantedContradiction:
    return PlantedContradiction(
        id=cid, issue=cid, question_id="Q1", side_a=a, side_b=b, magnitude=mag, kind=kind,
        correct_handling=handling, side_a_step=2, inject_step=INJECT, window="before_interim",
        relative_to="before_interim", recoverable=False, control_id="K" + cid,
    )


N = _c("N", "numeric", "resolve_by_rule", "choose_b",
       Claim("storage 41.7%", "41.7", "DOC-1", "equal"), Claim("storage 38.2%", "38.2", "DOC-2", "equal"))
D = _c("D", "direct", "must_preserve", "preserve_both",
       Claim("route Halcyne", "Halcyne", "DOC-3", "equal"), Claim("route Brosk", "Brosk", "DOC-4", "equal"))
# Qualifier: side A hedged from a credible source (correct), side B certain from a low-credibility source.
Q = _c("Q", "qualifier", "resolve_by_credibility", "choose_a",
       Claim("expected to exceed 126.5 ML/day, only if the Zornan upgrade", "126.5", "DOC-5", "high", "hedged", "Zornan"),
       Claim("will exceed 126.5 ML/day; confirmed", "126.5", "DOC-6", "low", "certain", None))

K_NUM = ControlFact("KN", "ctl", "Q1", Claim("55.3%", "55.3", "DOC-7", "high"), Claim("55.3%", "55.3", "DOC-8", "high"),
                    "numeric", 2, INJECT, False, "N")
K_QUAL = ControlFact("KQ", "ctl", "Q1",
                     Claim("likely to exceed 77.1 ML/day if the Velth deal", "77.1", "DOC-9", "high", "hedged", "Velth"),
                     Claim("likely to exceed 77.1 ML/day if the Velth deal", "77.1", "DOC-10", "high", "hedged", "Velth"),
                     "qualifier", 2, INJECT, False, "Q")

ENV_DOCS = (
    "The Supply Outlook (Edition 4) puts the storage level at 41.7%.\n\n"
    "The Supply Outlook (Edition 5) puts the storage level at 38.2%.\n\n"
    "Marrow study: the route is the Halcyne Pass route.\n\n"
    "Ostrand study: the route is the Brosk Pass route.\n\n"
    "The Audit Office states output is expected to exceed 126.5 ML/day, but only if the Zornan upgrade is completed.\n\n"
    "A forum post states output will exceed 126.5 ML/day; this figure is confirmed and carries no conditions."
)


def world() -> World:
    return World(seed=0, config=WorldConfig(), authority="Test Authority", brief="brief", questions=[],
                 documents=[], contradictions=[N, D, Q], controls=[K_NUM, K_QUAL])


def run(*steps_segments):
    """Run the probe over consecutive steps starting at INJECT; return list of StepResults."""
    p = PassiveProbe(world())
    return [p.step(INJECT + i, segs) for i, segs in enumerate(steps_segments)]


def label(results, cid, i=-1):
    return results[i].contradictions[cid].label


# ----------------------------------------------------------------- regexes

def test_number_boundaries():
    rx = marker_regex("41.7")
    assert rx.search("at 41.7% now")
    assert not rx.search("at 141.7%")
    assert not rx.search("at 41.75%")
    assert rx.search("the level is 41.7.")  # sentence-final period must still match


def test_name_whole_word_any_case():
    rx = marker_regex("Halcyne")
    assert rx.search("the HALCYNE route")
    assert not rx.search("Halcynes")


# ----------------------------------------------------------------- labels

def test_pre_injection():
    p = PassiveProbe(world())
    r = p.step(INJECT - 1, [Segment("env", ENV_DOCS)])
    assert r.contradictions["N"].label == "pre_injection"


def test_undetected_when_only_raw_docs():
    r = run([Segment("brief", "brief"), Segment("env", ENV_DOCS)])
    for cid in ("N", "D", "Q"):
        assert label(r, cid) == "undetected"


def test_held_both_when_linked_in_note():
    note = "Storage: Edition 4 says 41.7%, Edition 5 says 38.2%. Conflict, check which applies."
    r = run([Segment("env", ENV_DOCS), Segment("agent", note)])
    assert label(r, "N") == "held_both"
    assert r[-1].contradictions["N"].flagged


def test_linked_without_dispute_word_is_held_both():
    note = "Storage figures: 41.7% and 38.2%."
    r = run([Segment("agent", note)])
    assert label(r, "N") == "held_both"


def test_resolved_correct_superseded_by():
    note = "Storage 41.7% (Edition 4) superseded by 38.2% (Edition 5)."
    r = run([Segment("env", ENV_DOCS), Segment("agent", note)])
    assert label(r, "N") == "resolved_correct"


def test_resolved_wrong():
    note = "Storage: use 41.7%. The 38.2% figure is outdated."
    r = run([Segment("agent", note)])
    assert r[-1].contradictions["N"].resolution_seen == "a"
    assert label(r, "N") == "resolved_wrong"


def test_table_status_cells_resolve():
    table = "| Q1 | storage 41.7% | DOC-1 | superseded |\n| Q1 | storage 38.2% | DOC-2 | settled |"
    r = run([Segment("agent", table)])
    assert label(r, "N") == "resolved_correct"


def test_resolution_is_sticky_after_compaction():
    note = "Storage 41.7% (Edition 4) superseded by 38.2% (Edition 5)."
    summary = "Storage level is 38.2% per the latest edition."
    r = run([Segment("agent", note)], [Segment("summary", summary)])
    assert label(r, "N", 0) == "resolved_correct"
    assert label(r, "N", 1) == "resolved_correct"


def test_collapsed_without_resolution():
    r = run([Segment("summary", "Storage level is 41.7%.")])
    assert label(r, "N") == "collapsed_to_a"
    r = run([Segment("summary", "Storage level is 38.2%.")])
    assert label(r, "N") == "collapsed_to_b"


def test_collapse_to_rejected_side_after_resolution_is_not_resolved():
    note = "Storage 41.7% (Edition 4) superseded by 38.2% (Edition 5)."
    r = run([Segment("agent", note)], [Segment("summary", "Storage level is 41.7%.")])
    assert label(r, "N", 1) == "collapsed_to_a"


def test_lost():
    r = run([Segment("summary", "Nothing relevant here.")])
    for cid in ("N", "D", "Q"):
        assert label(r, cid) == "lost"


def test_reopened_dispute_clears_resolution():
    first = "Storage 41.7% (Edition 4) superseded by 38.2% (Edition 5)."
    second = "Storage: 41.7% vs 38.2%, disputed, the editions may cover different scopes."
    r = run([Segment("agent", first)], [Segment("agent", second)])
    assert label(r, "N", 1) == "held_both"


def test_must_preserve_held_and_wrongly_resolved():
    r = run([Segment("agent", "Route: Halcyne (Marrow) vs Brosk (Ostrand), disputed.")])
    assert label(r, "D") == "held_both"
    r = run([Segment("agent", "Route: go with Halcyne. Brosk claim is unreliable.")])
    assert label(r, "D") == "resolved_wrong"


def test_pinned_block_counts_as_agent_state():
    pinned = "Claims on the same issue from different sources:\n- 41.7% (DOC-1)\n- 38.2% (DOC-2)"
    r = run([Segment("pinned", pinned)])
    assert label(r, "N") == "held_both"


# ----------------------------------------------------------------- qualifier

def test_qualifier_side_detection():
    hedged, certain = Q.side_a, Q.side_b
    assert side_spans("output 126.5 ML/day only if the Zornan upgrade", hedged, "qualifier", certain)
    assert not side_spans("output 126.5 ML/day only if the Zornan upgrade", certain, "qualifier", hedged)
    assert side_spans("output will exceed 126.5 ML/day.", certain, "qualifier", hedged)
    assert not side_spans("output will exceed 126.5 ML/day.", hedged, "qualifier", certain)
    # condition token alone marks the hedged side
    assert side_spans("depends on Zornan", hedged, "qualifier", certain)


def test_qualifier_month_may_is_not_a_hedge():
    assert side_spans("As of May 2027 output will exceed 126.5 ML/day.", Q.side_b, "qualifier", Q.side_a)


def test_qualifier_unconditional_is_not_hedged():
    assert side_spans("Output 126.5 ML/day, unconditional.", Q.side_b, "qualifier", Q.side_a)


def test_qualifier_dropped_hedge_is_collapse_to_certain():
    r = run([Segment("summary", "Plant output: 126.5 ML/day.")])
    assert label(r, "Q") == "collapsed_to_b"


def test_qualifier_kept_hedge_is_collapse_to_hedged():
    r = run([Segment("summary", "Plant output expected above 126.5 ML/day if the Zornan upgrade completes.")])
    assert label(r, "Q") == "collapsed_to_a"


def test_qualifier_linked_and_resolved():
    note = ("Audit Office: 126.5 ML/day only if the Zornan upgrade completes; "
            "forum claim of 126.5 ML/day unconditional (low credibility, rejected).")
    r = run([Segment("agent", note)])
    assert label(r, "Q") == "resolved_correct"


def test_qualifier_known_miss_paraphrase_without_anchor():
    """Documented limitation: if the agent describes the certain side without
    restating the anchor, the passive probe cannot link the two sides."""
    note = "Output 126.5 ML/day is conditional on the Zornan upgrade; the forum's firm claim is unreliable."
    r = run([Segment("env", ENV_DOCS), Segment("agent", note)])
    assert label(r, "Q") == "undetected"


# ----------------------------------------------------------------- controls

def test_control_presence_and_activity():
    p = PassiveProbe(world())
    r0 = p.step(INJECT - 1, [Segment("agent", "Level 55.3%.")])
    assert r0.controls["KN"].present and not r0.controls["KN"].active
    r1 = p.step(INJECT, [Segment("env", "unrelated")])
    assert r1.controls["KN"].active and not r1.controls["KN"].present


def test_control_qualifier_modality():
    r = run([Segment("agent", "Transfers likely to exceed 77.1 ML/day if the Velth deal closes.")])
    assert r[-1].controls["KQ"].agent_present and r[-1].controls["KQ"].modality_preserved is True
    r = run([Segment("summary", "Transfers: 77.1 ML/day.")])
    assert r[-1].controls["KQ"].agent_present and r[-1].controls["KQ"].modality_preserved is False


def test_agent_vs_env_presence():
    r = run([Segment("env", ENV_DOCS), Segment("agent", "Storage 41.7%.")])
    s = r[-1].contradictions["N"]
    assert s.present_a and s.present_b and s.agent_a and not s.agent_b


def test_probe_on_generated_world_runs():
    """Smoke check that the probe handles a real generated world and its raw documents."""
    from worlds.generator import generate_world

    w = generate_world(0)
    p = PassiveProbe(w)
    delivered = []
    for step in range(1, 41):
        delivered += [d.render() for d in w.deliveries(step)]
        r = p.step(step, [Segment("brief", w.brief), Segment("env", "\n\n".join(delivered))])
    labels = {s.label for s in r.contradictions.values()}
    assert labels == {"undetected"}, labels
    assert all(k.present for k in r.controls.values())


# ------------------------------------------------- regressions from the smoke test

def test_qualifier_hedge_does_not_leak_across_list_items():
    """Smoke test 2026-11-08: a long comma list put another item's hedge in the
    same sentence as a certain anchor."""
    text = ("Supply: the desalination plant outputs >126.5 ML/day (DOC-76), "
            "emergency transfers likely >81.1 ML/day conditional on the Drymard agreement (DOC-694).")
    assert side_spans(text, Q.side_b, "qualifier", Q.side_a)  # certain
    assert not side_spans(text, Q.side_a, "qualifier", Q.side_b)  # not hedged


def test_qualifier_trailing_hedge_connective_after_comma():
    text = "Output will exceed 126.5 ML/day, but only if the upgrade completes."
    assert side_spans(text, Q.side_a, "qualifier", Q.side_b)
    assert not side_spans(text, Q.side_b, "qualifier", Q.side_a)


def test_qualifier_leading_hedge_connective():
    text = "Subject to the upgrade, output will exceed 126.5 ML/day."
    assert side_spans(text, Q.side_a, "qualifier", Q.side_b)
    assert not side_spans(text, Q.side_b, "qualifier", Q.side_a)


def test_qualifier_condition_token_in_sentence_blocks_certain():
    text = "Output 126.5 ML/day, per the Zornan upgrade plan."
    assert not side_spans(text, Q.side_b, "qualifier", Q.side_a)


def test_current_figure_phrasing_resolves():
    """Active probe answer, smoke test 2026-11-08."""
    text = ("The rate is 41.7% in Edition 4 and 38.2% in Edition 5. Since Edition 5 supersedes "
            "Edition 4, the current reliable figure is 38.2%.")
    r = run([Segment("agent", text)])
    assert label(r, "N") == "resolved_correct"


def test_qualifier_table_cell_parenthetical_hedge():
    """Smoke test 2026-11-08 regression: '<17.3/km (Ed7, conditional)' is the hedged side."""
    cell = "| Q3 | Zoutheth defect rate <126.5/km (Ed7, conditional) | DOC-791 | Superseded |"
    assert side_spans(cell, Q.side_a, "qualifier", Q.side_b)
    assert not side_spans(cell, Q.side_b, "qualifier", Q.side_a)


def test_qualifier_leading_hedge_on_next_item_does_not_attach_backwards():
    text = "Plant A output 126.5 ML/day, plant B likely 81.1 ML/day."
    assert side_spans(text, Q.side_b, "qualifier", Q.side_a)
    assert not side_spans(text, Q.side_a, "qualifier", Q.side_b)
