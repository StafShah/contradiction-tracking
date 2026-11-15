from probes.active import label_contradiction_answer, label_control_answer
from tests.test_passive_probe import K_NUM, N


def test_dont_know_variants_do_not_need_judge():
    for ans in [
        "I do not have current information regarding the storage level.",
        "I currently have no specific data regarding the repair backlog.",
        "I have not encountered any documents detailing this.",
    ]:
        f = label_contradiction_answer(N, ans)
        assert f["says_dont_know"] and not f["needs_judge"], ans


def test_marker_answers_labeled_without_judge():
    f = label_contradiction_answer(N, "Storage is 41.7% per Edition 4, but Edition 5 gives 38.2%; conflict.")
    assert f["label"] == "held_both" and not f["needs_judge"]
    f = label_control_answer(K_NUM, "The level is 55.3%.")
    assert f["label"] == "present"


def test_unmarked_substantive_answer_needs_judge():
    f = label_contradiction_answer(N, "Storage is somewhere around forty percent according to the outlook.")
    assert f["needs_judge"] and f["label"] == "lost"
