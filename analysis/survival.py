"""Pilot survival analysis.

    uv run python -m analysis.survival runs/<batch_dir> [--out analysis/figures]

Reads the *_rescored files (run analysis.rescore first) and produces:
- Kaplan-Meier curves of "correctly held" survival by policy, and by magnitude
  within each policy, with matched-control curves for comparison
- per-step hazard at post-compaction steps vs ordinary steps
- collapse direction by timing window (anchoring descriptives)
- final-memo outcomes and active-probe states
- results.json with every number used in analysis/pilot_results.md

Event definitions (decided before looking at pilot results; see NOTES.md):
- Contradiction failure: first step t >= inject_step whose passive label is
  `lost`, `resolved_wrong`, or a collapse that discards needed information
  (any collapse for must_preserve; collapse to the wrong side for resolve kinds).
  Collapse to the correct side of a resolve item is not a failure: the right
  answer is still held, just without the losing side.
- Control failure: first step t >= second_step where the control's marker is
  absent from the whole context.
- Time origin: inject_step (side B arrival) for contradictions, second_step for
  controls (the same step by construction). Censored at the final step.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.statistics import logrank_test  # noqa: E402

from analysis.rescore import load_world  # noqa: E402

POLICY_ORDER = ["full_context", "llm_summary", "oracle_pin"]
MAG_ORDER = ["direct", "numeric", "qualifier"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference categorical slots 1-3, validated (light)
STYLES = ["-", "--", ":"]  # secondary encoding: slot 3 is below 3:1 contrast
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def failed(label: str, handling: str) -> bool:
    if label in ("lost", "resolved_wrong"):
        return True
    if label.startswith("collapsed_to_"):
        side = label[-1]
        if handling == "preserve_both":
            return True
        return not ((side == "a" and handling == "choose_a") or (side == "b" and handling == "choose_b"))
    return False


def load_runs(batch: Path):
    runs = []
    for d in sorted(p for p in batch.iterdir() if p.is_dir()):
        if not (d / "passive_rescored.jsonl").exists():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        result = json.loads((d / "result.json").read_text(encoding="utf-8"))
        world = load_world(d / "world.json")
        passive = [json.loads(l) for l in (d / "passive_rescored.jsonl").open(encoding="utf-8")]
        active = [json.loads(l) for l in (d / "active_rescored.jsonl").open(encoding="utf-8")] if (d / "active_rescored.jsonl").exists() else []
        final = json.loads((d / "final_rescored.json").read_text(encoding="utf-8"))
        runs.append(dict(dir=d, policy=meta["policy"], seed=meta["world_seed"], world=world, passive=passive,
                         active=active, final=final, result=result))
    return runs


def durations(run) -> list[dict]:
    """One record per contradiction and per control: duration, event, covariates."""
    w = run["world"]
    last = w.config.final_step
    by_step = {p["step"]: p for p in run["passive"]}
    recs = []
    for c in w.contradictions:
        t_fail = None
        for t in range(c.inject_step, last + 1):
            lab = by_step[t]["contradictions"][c.id]["label"]
            if failed(lab, c.correct_handling):
                t_fail = t
                break
        recs.append(dict(item="contradiction", id=c.id, policy=run["policy"], seed=run["seed"], magnitude=c.magnitude,
                         kind=c.kind, window=c.window, inject=c.inject_step,
                         duration=(t_fail if t_fail else last) - c.inject_step, event=int(t_fail is not None), t_fail=t_fail))
    for k in w.controls:
        t_fail = None
        for t in range(k.second_step, last + 1):
            if not by_step[t]["controls"][k.id]["present"]:
                t_fail = t
                break
        recs.append(dict(item="control", id=k.id, policy=run["policy"], seed=run["seed"], magnitude=k.magnitude,
                         kind=None, window=None, inject=k.second_step,
                         duration=(t_fail if t_fail else last) - k.second_step, event=int(t_fail is not None), t_fail=t_fail))
    return recs


def km(recs):
    kmf = KaplanMeierFitter()
    kmf.fit([r["duration"] for r in recs], [r["event"] for r in recs])
    return kmf


def km_summary(recs, at=(5, 10, 20)) -> dict:
    if not recs:
        return {}
    kmf = km(recs)
    out = {"n": len(recs), "events": int(sum(r["event"] for r in recs))}
    ci = kmf.confidence_interval_survival_function_
    for t in at:
        s = float(kmf.survival_function_at_times(t).iloc[0])
        idx = ci.index[ci.index <= t].max()
        lo, hi = (float(ci.loc[idx].iloc[0]), float(ci.loc[idx].iloc[1])) if not np.isnan(idx) else (1.0, 1.0)
        out[f"S({t})"] = round(s, 3)
        out[f"S({t})_ci"] = [round(lo, 3), round(hi, 3)]
    return out


def hazard(recs, run_by_key, boundary_steps):
    """Pooled per-step hazard: events / at-risk, at post-compaction steps vs others."""
    at_risk = Counter()
    events = Counter()
    for r in recs:
        last = r["inject"] + r["duration"]
        for t in range(r["inject"], last + 1):
            key = "boundary" if t in boundary_steps else "ordinary"
            at_risk[key] += 1
            if r["event"] and t == r["t_fail"]:
                events[key] += 1
                break
    out = {}
    for key in ("boundary", "ordinary"):
        n, e = at_risk[key], events[key]
        out[key] = {"at_risk_steps": n, "events": e, "hazard": round(e / n, 4) if n else None}
    return out


def style_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("Steps since side B arrived", color=TEXT2, fontsize=8)


def place_labels(ax, items, min_gap=0.055):
    """Direct end labels, nudged apart vertically so they never collide."""
    items = sorted(items, key=lambda it: it[2], reverse=True)
    placed = []
    for name, x, y in items:
        y_lab = y
        for py in placed:
            if abs(py - y_lab) < min_gap:
                y_lab = py - min_gap
        placed.append(y_lab)
        ax.annotate(name, xy=(x, y), xytext=(x + 0.8, y_lab), textcoords="data", color=TEXT, fontsize=7.5, va="center")


def plot_curves(ax, groups: list[tuple[str, list]], title: str):
    style_axes(ax, title)
    labels = []
    for i, (name, recs) in enumerate(groups):
        if not recs:
            continue
        kmf = km(recs)
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_survival_function_
        x = sf.index.values
        ax.step(x, sf.iloc[:, 0].values, where="post", color=COLORS[i], linestyle=STYLES[i], linewidth=2)
        ax.fill_between(x, ci.iloc[:, 0].values, ci.iloc[:, 1].values, step="post", color=COLORS[i], alpha=0.12, linewidth=0)
        # direct label at the right end (identity is never color alone)
        labels.append((f"{name} (n={len(recs)})", x[-1], float(sf.iloc[-1, 0])))
    place_labels(ax, labels)
    ax.set_xlim(0, 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--out", default="analysis/figures")
    args = ap.parse_args()
    batch = Path(args.batch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(batch)
    invalid = [str(r["dir"].name) for r in runs if not r["result"].get("valid", True)]
    runs = [r for r in runs if r["result"].get("valid", True)]
    recs = [x for r in runs for x in durations(r)]
    con = [r for r in recs if r["item"] == "contradiction"]
    ctl = [r for r in recs if r["item"] == "control"]
    comp = runs[0]["world"].config.compaction_steps
    boundary = {s + 1 for s in comp}  # first step that sees the compacted context
    res: dict = {"batch": batch.name, "n_runs": len(runs), "invalid_runs": invalid,
                 "runs": Counter(r["policy"] for r in runs), "boundary_steps": sorted(boundary)}

    # ---- KM by policy, contradictions vs controls
    res["km_policy"] = {p: {"contradiction": km_summary([r for r in con if r["policy"] == p]),
                            "control": km_summary([r for r in ctl if r["policy"] == p])} for p in POLICY_ORDER}
    lr = {}
    for p in POLICY_ORDER:
        a = [r for r in con if r["policy"] == p]
        b = [r for r in ctl if r["policy"] == p]
        if a and b:
            t = logrank_test([r["duration"] for r in a], [r["duration"] for r in b], [r["event"] for r in a], [r["event"] for r in b])
            lr[f"{p}: contradiction vs control"] = round(float(t.p_value), 4)
    for p1, p2 in [("full_context", "llm_summary"), ("llm_summary", "oracle_pin")]:
        a = [r for r in con if r["policy"] == p1]
        b = [r for r in con if r["policy"] == p2]
        t = logrank_test([r["duration"] for r in a], [r["duration"] for r in b], [r["event"] for r in a], [r["event"] for r in b])
        lr[f"contradictions: {p1} vs {p2}"] = round(float(t.p_value), 4)
    res["logrank_p"] = lr

    # ---- magnitude within llm_summary (the failure condition), and kind
    for field, order in (("magnitude", MAG_ORDER), ("kind", ["resolve_by_rule", "resolve_by_credibility", "must_preserve"]),
                         ("window", ["before_interim", "after_interim_pre_compaction", "after_compaction"])):
        res[f"km_llm_summary_by_{field}"] = {v: km_summary([r for r in con if r["policy"] == "llm_summary" and r[field] == v]) for v in order}
    res["km_llm_summary_controls_by_magnitude"] = {m: km_summary([r for r in ctl if r["policy"] == "llm_summary" and r["magnitude"] == m]) for m in MAG_ORDER}

    # ---- never co-present: failed at the step side B arrived (side A already gone)
    res["failed_at_arrival"] = {p: sum(1 for r in con if r["policy"] == p and r["event"] and r["duration"] == 0) for p in POLICY_ORDER}
    res["km_policy_excluding_failed_at_arrival"] = {
        p: km_summary([r for r in con if r["policy"] == p and not (r["event"] and r["duration"] == 0)]) for p in POLICY_ORDER}

    # ---- prevalence by absolute step: fraction of delivered items currently in a non-failed state
    prev = {}
    for p in POLICY_ORDER:
        prs = [r for r in runs if r["policy"] == p]
        cs, ks = {}, {}
        for t in range(1, prs[0]["world"].config.final_step + 1) if prs else []:
            okc, okk = [], []
            for r in prs:
                row = r["passive"][t - 1]
                for c in r["world"].contradictions:
                    if c.inject_step <= t:
                        okc.append(not failed(row["contradictions"][c.id]["label"], c.correct_handling))
                for k in r["world"].controls:
                    if k.second_step <= t:
                        okk.append(row["controls"][k.id]["present"])
            cs[t] = round(float(np.mean(okc)), 3) if okc else None
            ks[t] = round(float(np.mean(okk)), 3) if okk else None
        prev[p] = {"contradiction": cs, "control": ks}
    res["prevalence_by_step"] = prev

    # ---- hazard at compaction boundaries
    res["hazard"] = {p: {"contradiction": hazard([r for r in con if r["policy"] == p], None, boundary),
                         "control": hazard([r for r in ctl if r["policy"] == p], None, boundary)} for p in POLICY_ORDER}

    # ---- failure label at the failure step, by policy
    fail_types = defaultdict(Counter)
    for r in runs:
        by_step = {p["step"]: p for p in r["passive"]}
        for d in durations(r):
            if d["item"] == "contradiction" and d["event"]:
                fail_types[r["policy"]][by_step[d["t_fail"]]["contradictions"][d["id"]]["label"]] += 1
    res["failure_labels"] = {p: dict(fail_types[p]) for p in POLICY_ORDER}

    # ---- final step label distribution
    final_labels = defaultdict(Counter)
    for r in runs:
        last = r["passive"][-1]
        for cid, v in last["contradictions"].items():
            final_labels[r["policy"]][v["label"]] += 1
    res["passive_labels_at_final_step"] = {p: dict(final_labels[p]) for p in POLICY_ORDER}

    # ---- collapse direction by window (anchoring descriptive). Side A is the only side
    # available at the interim step for after_interim items.
    direction = defaultdict(Counter)
    for r in runs:
        if r["policy"] != "llm_summary":
            continue
        last = r["passive"][-1]["contradictions"]
        for c in r["world"].contradictions:
            lab = last[c.id]["label"]
            if lab.startswith("collapsed_to_"):
                direction[c.window][lab] += 1
    res["llm_summary_collapse_direction_at_step40"] = {w: dict(v) for w, v in direction.items()}

    # ---- final memo outcomes
    fo = defaultdict(Counter)
    ctl_recall = defaultdict(list)
    for r in runs:
        for s in r["final"]["scores"]:
            if s["item_type"] == "contradiction":
                fo[r["policy"]][s["outcome"]] += 1
            else:
                ctl_recall[r["policy"]].append(s["outcome"] == "correct")
    res["final_outcomes"] = {p: dict(fo[p]) for p in POLICY_ORDER}
    res["final_contradiction_accuracy"] = {p: round(fo[p]["correct"] / max(1, sum(fo[p].values())), 3) for p in POLICY_ORDER}
    res["final_accuracy_given_mentioned"] = {
        p: round(fo[p]["correct"] / max(1, sum(v for k, v in fo[p].items() if k != "omitted")), 3) for p in POLICY_ORDER}
    res["final_control_recall"] = {p: round(float(np.mean(ctl_recall[p])), 3) if ctl_recall[p] else None for p in POLICY_ORDER}

    # ---- active probe: at each probe step, fraction of contradiction answers in a non-failed state,
    # and control answers that state the fact
    ap_ok = defaultdict(lambda: defaultdict(list))
    ap_judge = Counter()
    ap_total = Counter()
    for r in runs:
        handling = {c.id: c.correct_handling for c in r["world"].contradictions}
        for a in r["active"]:
            ap_total[r["policy"]] += 1
            ap_judge[r["policy"]] += int(a["needs_judge"])
            if a["item_type"] == "contradiction":
                ap_ok[r["policy"]][("c", a["step"])].append(not failed(a["label"], handling[a["item_id"]]))
            else:
                ap_ok[r["policy"]][("k", a["step"])].append(a["label"] == "present")
    res["active_probe"] = {
        p: {"needs_judge_frac": round(ap_judge[p] / max(1, ap_total[p]), 3),
            "contradiction_ok_by_step": {s: round(float(np.mean(v)), 3) for (kind, s), v in sorted(ap_ok[p].items()) if kind == "c"},
            "control_present_by_step": {s: round(float(np.mean(v)), 3) for (kind, s), v in sorted(ap_ok[p].items()) if kind == "k"}}
        for p in POLICY_ORDER}

    # ---- know-but-don't-output: at step 40, active probe answer OK but final memo outcome not correct
    kbdo = defaultdict(Counter)
    for r in runs:
        handling = {c.id: c.correct_handling for c in r["world"].contradictions}
        probe40 = {a["item_id"]: a for a in r["active"] if a["step"] == r["world"].config.final_step and a["item_type"] == "contradiction"}
        for s in r["final"]["scores"]:
            if s["item_type"] != "contradiction" or s["id"] not in probe40:
                continue
            probe_ok = not failed(probe40[s["id"]]["label"], handling[s["id"]]) and probe40[s["id"]]["label"] != "lost"
            kbdo[r["policy"]][f"probe_{'ok' if probe_ok else 'bad'}__final_{s['outcome']}"] += 1
    res["probe40_vs_final"] = {p: dict(kbdo[p]) for p in POLICY_ORDER}

    # ---- cost
    res["tokens_per_run"] = {p: {k: int(np.mean([r["result"][k] for r in runs if r["policy"] == p]))
                                 for k in ("tokens_in", "tokens_out", "probe_tokens_in", "probe_tokens_out", "summary_tokens_in")}
                             for p in POLICY_ORDER if any(r["policy"] == p for r in runs)}
    res["wall_seconds_per_run"] = {p: round(float(np.mean([r["result"]["wall_seconds"] for r in runs if r["policy"] == p])), 1) for p in POLICY_ORDER}
    res["behavior"] = {p: {"write_note_per_run": float(np.mean([sum(1 for _ in json.loads((r["dir"] / "final.json").read_text(encoding="utf-8"))["notes"]) for r in runs if r["policy"] == p])),
                           "forced_retries": sum(len(r["result"].get("forced_retries", [])) for r in runs if r["policy"] == p),
                           "early_final_attempts": sum(len(json.loads((r["dir"] / "final.json").read_text(encoding="utf-8")).get("early_final_attempts", [])) for r in runs if r["policy"] == p)}
                       for p in POLICY_ORDER}

    # ---- fairness check: a must_preserve item needs two facts to survive, a control needs one.
    # Under independence, the two-fact baseline is S_control(t)^2.
    fair = {}
    for p in ("llm_summary", "oracle_pin"):
        k20 = res["km_policy"][p]["control"].get("S(20)")
        fair[p] = {"control_S20": k20, "two_fact_baseline_S20": round(k20 ** 2, 3) if k20 is not None else None,
                   "must_preserve_S20": km_summary([r for r in con if r["policy"] == p and r["kind"] == "must_preserve"]).get("S(20)"),
                   "resolve_kinds_S20": km_summary([r for r in con if r["policy"] == p and r["kind"] != "must_preserve"]).get("S(20)")}
    res["fairness_two_fact_baseline"] = fair

    per_run = []
    for r in runs:
        d = [x for x in durations(r) if x["item"] == "contradiction"]
        k = [x for x in durations(r) if x["item"] == "control"]
        per_run.append({"run": r["dir"].name, "policy": r["policy"], "seed": r["seed"],
                        "contradictions_failed": sum(x["event"] for x in d), "controls_failed": sum(x["event"] for x in k),
                        "final_accuracy": r["final"]["summary"]["contradiction_accuracy"],
                        "final_control_recall": r["final"]["summary"]["control_recall"],
                        "write_notes": len(json.loads((r["dir"] / "final.json").read_text(encoding="utf-8"))["notes"])})
    res["per_run"] = per_run

    (out / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")

    # ---- figures
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    plot_curves(axes[0], [(p, [r for r in con if r["policy"] == p]) for p in POLICY_ORDER], "Contradictions correctly held, by memory policy")
    plot_curves(axes[1], [(p, [r for r in ctl if r["policy"] == p]) for p in POLICY_ORDER], "Matched control facts present, by memory policy")
    axes[0].set_ylabel("Fraction surviving (Kaplan-Meier, 95% CI)", color=TEXT2, fontsize=8)
    for ax in axes:
        ax.set_xlim(0, 46)
    fig.tight_layout()
    fig.savefig(out / "survival_by_policy.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    plot_curves(axes[0], [(m, [r for r in con if r["policy"] == "llm_summary" and r["magnitude"] == m]) for m in MAG_ORDER],
                "llm_summary: contradictions by magnitude")
    plot_curves(axes[1], [(m, [r for r in ctl if r["policy"] == "llm_summary" and r["magnitude"] == m]) for m in MAG_ORDER],
                "llm_summary: matched controls by magnitude")
    axes[0].set_ylabel("Fraction surviving (Kaplan-Meier, 95% CI)", color=TEXT2, fontsize=8)
    for ax in axes:
        ax.set_xlim(0, 46)
    fig.tight_layout()
    fig.savefig(out / "survival_llm_summary_by_magnitude.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, item, title in ((axes[0], "contradiction", "Contradictions in a correct state, by absolute step"),
                            (axes[1], "control", "Control facts present, by absolute step")):
        style_axes(ax, title)
        ax.set_xlabel("Agent step", color=TEXT2, fontsize=8)
        for s_ in comp:
            ax.axvline(s_ + 0.5, color=TEXT2, linewidth=1, linestyle=(0, (2, 2)))
            ax.annotate("compaction", xy=(s_ + 0.5, 0.02), xytext=(3, 0), textcoords="offset points", color=TEXT2, fontsize=7)
        labels = []
        for i, p in enumerate(POLICY_ORDER):
            series = prev.get(p, {}).get(item, {})
            xs = [t for t, v in series.items() if v is not None]
            ys = [series[t] for t in xs]
            if not xs:
                continue
            ax.plot(xs, ys, color=COLORS[i], linestyle=STYLES[i], linewidth=2)
            labels.append((p, xs[-1], ys[-1]))
        place_labels(ax, labels)
        ax.set_xlim(1, 47)
    axes[0].set_ylabel("Fraction of delivered items (pooled over seeds)", color=TEXT2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "prevalence_by_step.png", dpi=160)
    plt.close(fig)

    print(json.dumps({k: v for k, v in res.items() if k != "prevalence_by_step"}, indent=1, default=str))


if __name__ == "__main__":
    main()
