# CLAUDE.md: Contradiction Tracking in Long-Horizon LLM Agents

This file is the full handoff context for this repo. Read it completely before writing code.
Last updated: 2026-11-14.

---

## 1. Who and why

- Owner: Mustafa Shah. OMSCS student at Georgia Tech (ML specialization), software engineer at Roblox, builds agentic systems professionally.
- Hardware: Windows desktop with an NVIDIA RTX 5090 (32 GB VRAM). Open models run locally. Modest frontier API spend is self-funded.
- Goal: a rigorous, publishable research project. Primary target is a workshop paper (NeurIPS / ICLR / ICML / COLM agent, memory, or evaluation workshops), stretch target ACL/EMNLP via ACL Rolling Review.
- Academic track: CS 6999 (9-hour MS project). Needs a College of Computing faculty supervisor; outreach is in progress (see section 10). The work proceeds whether or not a supervisor signs on.

### Writing and working preferences

- Never use em-dashes in any prose, docs, comments, or paper text.
- Prefer concise, direct, honest analysis. Flag weak results plainly. Do not oversell findings.
- Ground truth must be deterministic wherever possible. LLM judges only where unavoidable, and always validated against hand labels.

---

## 2. Research question

When an LLM agent encounters a contradiction partway through a long, multi-step task, does correct handling of that contradiction survive the rest of the task? Specifically: how long does it stay represented, does it get resolved correctly, preserved when it should be, or silently collapsed or lost, and which events (context compaction, forming an interim conclusion) drive the failure?

Then: propose and evaluate fixes targeted at wherever the failure is localized.

### Research questions (from proposal draft 1)

1. Once detected, how long does a contradiction remain represented in the agent's working state?
2. Do slight contradictions (a removed hedge, a small numeric discrepancy) disappear earlier and more silently than direct ones?
3. Are contradictions arriving after the agent forms an interim conclusion resolved in favor of that conclusion more often than matched ones arriving earlier?
4. Does the agent's decision to resolve versus preserve a conflict remain correct across steps?
5. Does an intervention change these survival curves? (Updated: the intervention is now online detection plus a two-sided conflict ledger, compared against pinning baselines. See section 5.)

### Falsifiable hypotheses

- **H1 (magnitude):** qualifier and numeric contradictions are lost faster than direct ones. Falsified if survival curves overlap within CIs.
- **H2 (compaction):** the per-step hazard of losing a contradiction spikes at compaction boundaries. Falsified if hazard at boundary steps is about equal to non-boundary steps.
- **H3 (compaction-laundered confirmation bias):** contradictions that arrive after an interim conclusion collapse toward that conclusion, and this worsens after compaction because the summary keeps the conclusion and drops the counterevidence. Falsified if collapse direction is about 50/50.
- **H4 (pinning is not enough):** constraint-style pinning protects rules but not contradictions; naive pinning of one side may increase one-sided collapse. Falsified if naive pinning matches the conflict ledger.

If contradictions decay uniformly regardless of magnitude, kind, and timing, the finding reduces to generic long-horizon forgetting. Fallback contribution: per-stage localization of where loss happens (note-taking, compaction, planning, final synthesis).

---

## 3. Literature map (what exists, and how we differ)

Verify exact claims against the papers before citing in the paper. Items marked (abstract-level) were read only at abstract or secondary-source depth.

### Closest prior work

| Work | What it does | Relation to us |
|---|---|---|
| **KC-Bench** (Lyu et al., arXiv 2609.03588, Sep 2026). Code: github.com/ASTAR123/KC-Bench | 238 interactive tasks, tau-bench style (user simulator, stateful tools). Three conflict types: world-knowledge (false user premise), input inconsistency (forged credential), multi-source temporal (expired vs active record). Conflicts present from the start, short trajectories (avg 1.3 to 10.9 turns, cap 30). Correct behavior is always resolution (reject, reconcile, clarify). Binary per-episode LLM-judge scoring with human verification. No compaction. Finds no model reliable; notes "temporal sycophancy" (GPT-5.1 abandons a correct conclusion under user insistence). | Closest topical work. We add: mid-trajectory injection, long horizons, compaction, resolve-vs-preserve, per-step survival. Candidate environment base to fork. Candidate collaborator. |
| **Governance Decay** (Shiyang Chen, arXiv 2606.22528, Jun 2026) (abstract-level plus secondary writeups) | ConstraintRot benchmark, 1,323 episodes, 7 model families, deterministic tool-call grading. Prohibitions given up front. Violations 0% full context, 30% after compaction (up to 59%). Constraint survives summary: 0% violation; dropped: 38%. Compaction-Eviction Attack biases summarizer to drop policy. **Constraint Pinning**: keep policy (~47 tokens) out of compaction, re-inject verbatim after each compaction; restores 0% violation, ~1% over-refusal. Referee notes thin method detail; code release unconfirmed. | Owns "compaction erases constraints, pinning fixes it." Our original mitigation (a record compaction can't remove) is essentially this, so it is now a **baseline**, not our contribution. Pinning requires knowing a priori what to pin; contradictions are discovered mid-task, two-sided, and have evolving status. Candidate collaborator. |
| Companion to Governance Decay (arXiv 2604.20911, Apr 2026) (secondary-source only) | Omission ("never do X") constraints fade under context pressure; commission ("always do Y") persist. "Security-Recall Divergence." | Related framing. Verify before citing. |
| **Benchmarks Are Surprisingly Compaction-Friendly** (Salle, Bhatnagar, Bethge, Abdelnabi, Andriushchenko, arXiv 2610.31778, Oct 2026) | Compaction is lossy but benchmarks hide it because objectives are fixed and progress persists on disk. First summary drops about one of three anti-cheating rules on ProgramBench; process instructions dropped about 6x the rate of feature requirements on SWE-chat. Calls for tasks where constraints cannot be recovered from the filesystem. | Validates the direction. We must include a **recoverable vs non-recoverable** condition. |
| **Compact-Bench** (Sun et al., USTC, arXiv 2610.31710, Oct 2026) | Coding agents. Full-history vs summary-only continuation from same state. Best LLM summary retains 16% of recurring literal details; a mechanical extraction baseline retains 79.1%. Single boundary, final-outcome scoring, no per-step probing, no contradictions. | Gives us the **extractive memory baseline**. Literal-substring retention metric is a useful passive-probe precedent. |
| **What Does Context Compression Cost an Agent?** (Shuyu Liu, arXiv 2608.16370) | IRBench: hidden constraints revealed during execution. Compression barely changes completion but sharply raises retrieval (re-query) calls. | Track cost and re-query counts, not just success. |
| **Lost in Compaction** | Constraint loss within one compaction event. | Single-boundary; we track across many steps and boundaries. |
| **Facts Without Rules** | Metadata loss in a single handoff. | Single-handoff. |
| **Traverse** | First-error localization in agent trajectories. | Localization method precedent; not contradiction-specific. |
| **Stale Instruction Files Override the Code** (arXiv 2610.30571) | Coding agents followed an outdated instruction file over actual code in ~90% of choices, even when they explicitly noted the conflict. | Motivating statistic. Evidence for "know but don't act." |
| **DERELAB** (Sadhu, Shahad, Marino, arXiv 2608.30413) | Defeasible reasoning, multi-turn chat (not agentic). Defeaters arrive mid-conversation after the model commits. Confirmation bias near-universal (17 of 19 cells significant). "Know-but-don't-output": models label an update as weakening but still answer yes. Extended thinking reduces bias. | Covers RQ3-style anchoring but without tools, long horizons, or compaction. Borrow the congruent vs incongruent gap as our **anchoring index**. |
| **BayesBench** (Samanta et al., Meta, arXiv 2606.30850) | Per-turn belief trajectories from option log-probs vs Bayesian references. Not agentic. | Precedent for per-step belief measurement and log-prob probing on open models. |
| **AgentProcessBench**, **Agent Step Value** (arXiv 2607.04419), **What Process Evaluation of Coding Agents Actually Measures** (arXiv 2608.22960) | Step-level process evaluation of agents. | Per-step scoring is an established method, not our contribution. Cite as methodology. |
| **The Long-Horizon Task Mirage?** (Wang et al., arXiv 2604.11978, includes GT authors) | Horizon definitions, 7-category failure taxonomy, 3,100+ trajectories, LLM-judge attribution. | Taxonomy context. |
| **How Fast Do Agents Rot?** (Mittal, arXiv 2609.01660) | Long-horizon degradation; step count more explanatory than context length. | Generic degradation. We must show contradiction-specific effects beyond this. |
| **AMA-Bench** (Zhao et al., ICML 2026) | Long-horizon agent memory evaluation. | Memory eval context. |
| **Synthetic Worlds / Parallel Universes for Temporal Evaluation and Knowledge Updating in LLMs** (Wei Xu, Alan Ritter et al., arXiv 2609.00184, NeurIPS 2026) | Fictional but plausible worlds to avoid leakage and clashes with real facts. | Design precedent for our fictional document worlds. |
| **Memo** (Gupta, Yadav, Kira, Gal, Aljundi, arXiv 2510.19732, NeurIPS 2025) | Learned periodic summarization tokens for long-horizon embodied RL. | Learned compaction analog. |
| **Revisiting DAgger in the Era of LLM-Agents** (Chao Zhang's group, arXiv 2605.12913) | Early mistakes shift later state distribution. | Compounding-error framing. |
| **Position: AI Is Not Ready for Strategic Conflicts** (Riedl, Matlin, arXiv 2609.16189, Social Sim workshop at COLM 2026) | Decision laundering via summaries; early mistaken assumptions persist as state; loss of "strategic continuity"; summaries should retain uncertainty and sources. Does not address contradictory sources directly. | High-stakes motivation for the policy domain. |
| Single-step conflict benchmarks: **WikiContradict**, **RAMDocs** | Knowledge conflicts at one point in time. | Single-step. |
| Mechanistic and authority work: **How Language Models Choose Sides**, **Prompt Injection as Role Confusion**, **Authority Hierarchy in Sycophancy**, **Where Instruction Hierarchy Breaks**, **Agents' Overreliance on Unreliable Tools** | Authority vs veracity, source trust. | Secondary mechanistic arm (probes/steering on open models): is the truth signal erased or preserved and overridden? Not in pilot scope. |

### What is net new (our claimed contribution)

1. **Contradictions are relational, discovered mid-task, and have status.** Pinning only works for known, single-item, up-front rules. Online detection plus a two-sided ledger with provenance and status is an open design problem.
2. **Mid-trajectory injection at controlled timing** (before/after interim conclusion, before/after compaction) in long runs (30+ steps).
3. **Resolve vs preserve correctness.** Prior work scores loss or binary success only.
4. **Per-step survival analysis** (survival curves, hazard at compaction boundaries) instead of single end-of-episode scores.
5. **Specific hypotheses no prior paper can answer:** compaction-laundered confirmation bias (H3); temporal sycophancy accumulating over steps and compactions; naive pinning backfiring (H4).

### Do NOT claim

- That step-level trajectory evaluation is new.
- That compaction losing information is new.
- That pinning or protected memory regions are new.
- Anything about AI being categorically unsafe for diplomacy or high-stakes use.

---

## 4. Experimental design

### Design matrix (per planted contradiction)

- **magnitude:** `qualifier` (hedge removed or added), `numeric` (small discrepancy), `direct` (outright contradiction)
- **kind / correct handling:**
  - `resolve_by_rule` (e.g. newer record supersedes older; current code over stale docs)
  - `resolve_by_credibility` (one source is clearly more authoritative)
  - `must_preserve` (genuinely competing accounts; correct output keeps both, attributed)
- **timing:** `before_interim` / `after_interim` (relative to the agent's first interim conclusion), crossed with `before_compaction` / `after_compaction`
- **recoverable:** whether the evidence remains re-readable on disk later (true/false)

### Core schema

```python
@dataclass
class Claim:
    text: str
    marker: str          # unique planted token: entity name, number, date. Enables deterministic passive probing.
    source_id: str
    source_credibility: Literal["high", "low", "equal"]

@dataclass
class PlantedContradiction:
    id: str
    side_a: Claim
    side_b: Claim
    magnitude: Literal["qualifier", "numeric", "direct"]
    kind: Literal["resolve_by_rule", "resolve_by_credibility", "must_preserve"]
    correct_handling: Literal["choose_a", "choose_b", "preserve_both"]
    inject_step: int
    relative_to: Literal["before_interim", "after_interim", "before_compaction", "after_compaction"]
    recoverable: bool
```

### Per-step state labels (one per contradiction per step)

`undetected`, `held_both` (both sides present and linked as a conflict), `collapsed_to_a`, `collapsed_to_b`, `resolved_correct`, `resolved_wrong`, `lost` (neither side present).

### Pipeline

1. **World generation:** fictional document set (policy memo or consulting-style analysis) with N planted contradictions sampled from the design matrix. Fictional entities avoid parametric-knowledge leakage.
2. **Agent run:** one fixed ReAct-style agent with tools `read_doc`, `search`, `write_note`, `conclude_interim`, `final_answer`. Same model, prompt, and seed set across conditions. Only the memory policy varies. Compaction fires at fixed step indices so timing is controlled.
3. **Per-step snapshot:** full context window, notes, current summary, tool call, model output. Persist everything as JSONL.
4. **Passive probe (every step, free):** deterministic scan of the snapshot for each side's marker and source, and whether both sides are linked.
5. **Active probe (every K steps):** fork the context, ask "What is your current understanding of X and why?", label the answer, discard the fork so it never affects the main trajectory. Separates written state from retrievable belief.
6. **Labeling:** assign per-step state label per contradiction.
7. **Final scoring:** was each contradiction handled correctly in the final answer?
8. **Analysis:** Kaplan-Meier survival curves by magnitude, kind, timing, policy; hazard at compaction vs non-compaction steps; Cox regression with covariates.

### Memory policies (same interface, swappable)

| Policy | Role |
|---|---|
| `full_context` | Upper bound, no compaction |
| `sliding_window` | Naive eviction |
| `llm_summary` | Standard compaction (the failure condition) |
| `extractive` | Mechanical extraction baseline (Compact-Bench) |
| `oracle_pin` | Pin both sides of every ground-truth contradiction after injection. Ceiling: perfect detection |
| `naive_pin` | Pin whatever the agent noted first about each topic. Realistic pinning; possible backfire |
| `conflict_ledger` | **Our intervention:** online detection, structured record with both sides, provenance, status (open / resolved / preserved), excluded from compaction |

The oracle_pin vs naive_pin gap measures how much of the problem is detection vs retention.

### Metrics

Survival:
- Survival function S(t): fraction of contradictions still correctly held at step t
- Hazard at compaction boundary vs ordinary steps
- Recoverable vs non-recoverable gap

Failure modes:
- Detection rate and latency (steps from injection to detection)
- Resolve vs preserve confusion matrix
- Collapse rate and direction (toward interim conclusion vs away) = anchoring index
- Wrongful resolution rate (must_preserve conflicts that got resolved, and vice versa)
- Know-but-don't-act gap: active probe correct while action or final output wrong
- Failure localization: first stage where handling breaks (note-taking, compaction, planning, final synthesis)

Outcome and cost:
- Final handling accuracy, task success
- Tokens, latency, re-read or retrieval calls per run

Validation:
- LLM judge used only for active-probe answers; validate on about 200 hand-labeled samples and report agreement.

### Targeted fixes (phase after diagnosis)

| If failure is localized to | Proposed fix |
|---|---|
| Summarizer drops a side | Conflict-aware compaction / conflict ledger |
| Ledger holds it but decisions ignore it | Pre-decision check: query open conflicts before `conclude_interim`, `final_answer`, or any irreversible tool call |
| Anchoring after interim conclusion | Explicit "does this change my conclusion?" re-evaluation step on new related evidence |
| Know-but-don't-act | Output-time consistency check of final answer against ledger and probe answers |

---

## 5. Stack and infrastructure

- **Language:** Python 3.11+. Use `uv` for environments and dependencies.
- **Eval framework:** UK AISI **Inspect** (`inspect_ai`). Use it for tasks, solvers, tools, sandboxes, logging, and transcripts. Our new code is tasks, memory policies, probes, and scorers. Do not hand-roll an eval runner.
- **Environment base:** before building custom environments, clone and inspect `github.com/ASTAR123/KC-Bench`. If its environment and simulator can be extended within about a day to support long horizons, mid-run injection, and swappable compaction, build on it. Otherwise use a minimal synthetic document world for the pilot and port to KC-Bench in phase 2.
- **Local models:** the machine is Windows. Prefer vLLM inside WSL2 (CUDA) serving an OpenAI-compatible endpoint; Ollama is an acceptable fallback for the pilot. Confirm what is installed before assuming. Choose a current open instruct model with reliable tool calling that fits in 32 GB VRAM (roughly 7B to 32B, quantized if needed). Record exact model ID and quantization.
- **Frontier spot checks:** via API keys from environment variables only. Never commit keys.
- **Reproducibility:** log model ID and version, date, prompts, seeds, sampling params, compaction step indices, and full transcripts for every run. Every result in the paper must be regenerable from a config file plus the logs.

### Intended layout

```
contradiction-tracking/
  CLAUDE.md
  README.md
  pyproject.toml
  configs/              # experiment configs (yaml): model, policy, design-matrix sample, seeds
  worlds/
    generator.py        # fictional document sets + PlantedContradiction specs
    templates/
  tasks/                # Inspect task definitions (policy memo first; consulting and small coding arm later)
  agent/
    react_agent.py
    tools.py
    memory/
      base.py           # MemoryPolicy interface: observe(step), compact(step) -> context, pinned()
      full_context.py
      sliding_window.py
      llm_summary.py
      extractive.py
      oracle_pin.py
      naive_pin.py
      conflict_ledger.py
  probes/
    passive.py
    active.py
  scoring/
    per_step.py
    final.py
  analysis/
    survival.py
    notebooks/
  runs/                 # gitignored raw logs
  tests/
```

---

## 6. Pilot scope (do this first)

Purpose: test whether the phenomenon exists and is not uniform before investing in the full matrix.

Detailed, owner-approved pilot design: `PILOT_PLAN.md`. Dated decisions and smoke-test results: `NOTES.md`. Changes from the outline below (2026-11-08): the brief gives 6 broad questions instead of naming tracked issues; the pilot model is QuantTrio/Qwen3.6-35B-A3B-AWQ on vLLM; API spend is capped at $500 for the whole project.

- 1 task family: fictional policy-memo research task
- 1 local open model
- About 20 planted contradictions per world, spanning all magnitude x kind cells, with a subset of timing conditions
- 3 memory policies: `full_context`, `llm_summary`, `oracle_pin`
- Run length: 30 to 40 steps with 2 compaction events at fixed steps
- 5 seeds per condition
- Passive probe every step; active probe every 5 steps

### Go / no-go after pilot

- **Go:** `llm_summary` shows clearly lower survival than `full_context`, and either a compaction-boundary hazard spike or a magnitude/timing difference is visible.
- **Adjust:** everything survives under all policies. Increase horizon, distractor volume, or compaction aggressiveness before concluding.
- **Pivot:** decay is uniform across magnitude, kind, and timing. Shift the main contribution to per-stage localization.

### Pilot deliverables

- Working harness with all three policies
- Survival plot by policy and by magnitude
- A short markdown results note in `analysis/pilot_results.md`: what was run, numbers, caveats, next decision
- A GitHub repo link suitable for faculty and collaborator emails

---

## 7. Phase plan after pilot

1. Full design matrix, all 7 memory policies, 2 to 3 open models, frontier spot checks.
2. Add consulting-style task family and a small coding transfer arm.
3. Implement and evaluate the targeted fixes from section 4.
4. Judge validation on hand labels.
5. Optional mechanistic arm on open models: linear probes or steering to test whether the truth signal is erased or preserved and overridden.
6. Paper draft (IEEE or ACL format, Overleaf).

---

## 8. Threats to validity to design against

- Model vs harness confounds: hold model fixed, vary one component at a time.
- Recoverability confound (agents re-read files): explicit recoverable flag.
- Judge bias: deterministic markers first, judge only for active probes, validated.
- Probe contamination: active probes run in discarded forks only.
- Generic forgetting: compare contradiction survival against matched non-contradictory control facts planted at the same steps. This control is required to claim contradiction-specific effects.
- API drift: record versions and dates.

---

## 9. Proposal status

Proposal draft 1 exists ("Tracking Contradictions Through Long-Horizon LLM Agent Tasks"), saved as a PDF by the owner. It needs updating to reflect:
- KC-Bench and Governance Decay in the Gap section
- Intervention reframed: online detection + two-sided conflict ledger, evaluated against oracle and naive pinning
- Fill "[semester(s)]"; keep "A pilot is underway" only once true

---

## 10. Outreach status (context only, not code work)

- Advising: CS 6999 requires a CoC faculty supervisor; DocuSign form plus full proposal due before the first day of classes of the starting term; open a new oms-advising ticket after a supervisor agrees. OMSCS Associate Director of Research: Nick Lytle.
- Wave 1 emailed (around Oct 31 to Nov 4, 2026): Wei Xu, Chao Zhang, Mark Riedl, plus Theodore LaGrow (leads/guidance; likely not CoC-eligible) and Zsolt Kira (owner took his CS 7643). No replies yet; follow-ups planned week of Nov 16.
- Wave 2 held: Alan Ritter, Kartik Goyal (after follow-ups); Srijan Kumar (credibility angle).
- Fallback path: independent workshop paper; collaborate with KC-Bench and Governance Decay authors using pilot results.
- A pilot result plus repo link should be added to follow-up emails as soon as it exists.

---

## 11. Rules for Claude working in this repo

- Read this file first every session. Update the "Last updated" date and relevant sections when the plan changes.
- Keep a running `NOTES.md` log of decisions and results with dates.
- Small, testable steps. Write a unit test for the world generator and the passive probe before running anything expensive.
- Run a 1-seed, 1-world smoke test before any batch.
- Never commit API keys, raw run logs, or model weights.
- Follow the commit, tag, and experiment-workflow conventions in `README.md` ("Commit and versioning conventions"). Commit before launching any batch so its `meta.json` records the code version.
- Report results honestly, including null results.
- No em-dashes anywhere.
