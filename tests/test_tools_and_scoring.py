from agent.tools import DocEnvironment
from scoring.final import score_contradiction, score_memo, summarize
from tests.test_passive_probe import D, N, Q
from worlds.generator import generate_world


def test_transient_items_not_readable():
    w = generate_world(0)
    env = DocEnvironment(w)
    c = w.contradictions[0]
    r = env.execute(40, "read_doc", {"doc_id": c.side_a.source_id})
    assert r.is_error
    bg = next(d for d in w.documents if d.role == "background")
    assert not env.execute(1, "read_doc", {"doc_id": bg.doc_id}).is_error
    assert not env.execute(1, "read_doc", {"doc_id": bg.doc_id.lower()}).is_error


def test_search_only_library():
    w = generate_world(0)
    env = DocEnvironment(w)
    c = w.contradictions[0]
    r = env.execute(40, "search", {"query": c.issue})
    assert c.side_a.source_id not in r.text and c.side_b.source_id not in r.text


def test_unknown_tool_and_bad_args():
    env = DocEnvironment(generate_world(0))
    assert env.execute(1, "nope", {}).is_error
    assert env.execute(1, "read_doc", {"wrong": "x"}).is_error


def test_final_answer_done():
    env = DocEnvironment(generate_world(0))
    r = env.execute(40, "final_answer", {"memo": "m"})
    assert r.done and env.final_memo == "m"


def test_early_final_answer_rejected():
    env = DocEnvironment(generate_world(0))
    r = env.execute(39, "final_answer", {"memo": "m"})
    assert r.is_error and not r.done and env.final_memo is None
    assert env.early_final_attempts == [39]


# ------------------------------------------------------------------ scoring

def test_score_choose_b_correct():
    memo = "Q1 | storage 38.2% | DOC-2 | settled\nQ1 | storage 41.7% | DOC-1 | superseded"
    assert score_contradiction(N, memo).outcome == "correct"


def test_score_choose_b_both_unresolved_is_wrongly_preserved():
    memo = "Storage is 41.7% or 38.2%; sources disagree."
    assert score_contradiction(N, memo).outcome == "wrongly_preserved"


def test_score_wrong_choice():
    assert score_contradiction(N, "Storage is 41.7%.").outcome == "wrong_choice"


def test_score_omitted():
    assert score_contradiction(N, "No storage figures.").outcome == "omitted"


def test_score_must_preserve():
    assert score_contradiction(D, "Route: Halcyne (Marrow) vs Brosk (Ostrand), disputed.").outcome == "correct"
    assert score_contradiction(D, "Route: Halcyne.").outcome == "wrongly_resolved"
    assert score_contradiction(D, "Route: go with Halcyne; the Brosk claim is unreliable.").outcome == "wrongly_resolved"


def test_score_qualifier_keeps_hedge():
    memo = "Output expected to exceed 126.5 ML/day, only if the Zornan upgrade completes."
    assert score_contradiction(Q, memo).outcome == "correct"
    assert score_contradiction(Q, "Output will exceed 126.5 ML/day.").outcome == "wrong_choice"


def test_score_memo_and_summary_on_generated_world():
    w = generate_world(0)
    scores = score_memo(w, "")
    s = summarize(scores)
    assert s["contradiction_accuracy"] == 0 and s["control_recall"] == 0
    assert s["contradiction_outcomes"] == {"omitted": 20}
