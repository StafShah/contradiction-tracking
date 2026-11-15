"""Recompute every label for a finished run from its saved logs, with the
current probe and scoring code. No model calls. Use after any probe fix so all
runs are labeled by the same code version.

    uv run python -m analysis.rescore runs/<batch>/<sample> [more run dirs...]

Writes passive_rescored.jsonl, active_rescored.jsonl, final_rescored.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from probes.active import label_contradiction_answer, label_control_answer
from probes.passive import PassiveProbe, Segment
from scoring.final import score_memo, summarize, to_dicts
from worlds.schema import Claim, ControlFact, Document, PlantedContradiction, Question, World, WorldConfig


def load_world(path: Path) -> World:
    d = json.loads(path.read_text(encoding="utf-8"))
    cfg = d["config"]
    cfg["compaction_steps"] = tuple(cfg["compaction_steps"])
    cfg["windows"] = {k: tuple(v) for k, v in cfg["windows"].items()}
    return World(
        seed=d["seed"], config=WorldConfig(**cfg), authority=d["authority"], brief=d["brief"],
        questions=[Question(**q) for q in d["questions"]],
        documents=[Document(**x) for x in d["documents"]],
        contradictions=[PlantedContradiction(**{**c, "side_a": Claim(**c["side_a"]), "side_b": Claim(**c["side_b"])}) for c in d["contradictions"]],
        controls=[ControlFact(**{**k, "first": Claim(**k["first"]), "second": Claim(**k["second"])}) for k in d["controls"]],
    )


def rescore(run_dir: Path) -> dict:
    world = load_world(run_dir / "world.json")
    steps = [json.loads(l) for l in (run_dir / "steps.jsonl").open(encoding="utf-8")]
    steps = [s for s in steps if "event" not in s]

    probe = PassiveProbe(world)
    with (run_dir / "passive_rescored.jsonl").open("w", encoding="utf-8") as f:
        for s in steps:
            segs = [Segment(x["kind"], x["text"], x["step"]) for x in s["context"]]
            r = probe.step(s["step"], segs)
            f.write(json.dumps({"step": s["step"], "contradictions": {k: asdict(v) for k, v in r.contradictions.items()},
                                "controls": {k: asdict(v) for k, v in r.controls.items()}}) + "\n")

    items = {c.id: c for c in world.contradictions} | {k.id: k for k in world.controls}
    active_path = run_dir / "active.jsonl"
    if active_path.exists():
        with (run_dir / "active_rescored.jsonl").open("w", encoding="utf-8") as f:
            for line in active_path.open(encoding="utf-8"):
                a = json.loads(line)
                item = items[a["item_id"]]
                fields = label_contradiction_answer(item, a["answer"]) if a["item_type"] == "contradiction" else label_control_answer(item, a["answer"])
                f.write(json.dumps({**a, **fields}) + "\n")

    final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    scores = score_memo(world, final["memo"])
    out = {"scores": to_dicts(scores), "summary": summarize(scores)}
    (run_dir / "final_rescored.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out["summary"]


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(d, json.dumps(rescore(Path(d))))
