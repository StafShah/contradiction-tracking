# Pilot Plan

Status: draft for owner approval. Written 2026-11-08. Nothing in this plan has been run yet.

## 0. Inputs this plan is based on

### Environment (checked 2026-11-08)

| Item | Found | Notes |
|---|---|---|
| GPU | RTX 5090, 32,607 MiB, driver 610.88 (CUDA UMD 13.3) | ~0.7 GB used by the desktop |
| WSL2 | Ubuntu-24.04, WSL version 2, GPU visible inside WSL | 24 cores, 39 GB RAM, 934 GB free disk |
| Python (Windows) | 3.14.7 (system and uv-managed) | Will pin the project to 3.12 for wheel compatibility (inspect_ai, numpy, lifelines) |
| Python (WSL) | 3.12.3 system | No torch, no vLLM. One unrelated venv at ~/orbit/.venv |
| uv | 0.12.3 on Windows; **not installed in WSL** | |
| git | 2.55.0 on Windows | Repo is not yet a git repo |
| vLLM | not installed | |
| Ollama | not installed (Windows or WSL) | |
| CUDA toolkit (nvcc) in WSL | not installed | Not needed for prebuilt wheels |

### KC-Bench assessment (cloned to `../KC-Bench`, commit adf93de, 2026-07-28)

What it is: a fork of tau2-bench (the package is still named `tau2` in pyproject, module `kc`). 238 tasks across three domains (region 71, retail 93, personalAssistant 74). An orchestrator alternates agent, LLM user simulator, and environment. Conflicts are baked into the static DB or the user persona at t=0 (for example personalAssistant1: a stale phone number in the contact list vs a newer one in contact history). Scoring is hard-wired to an LLM judge over NL assertions (`evaluator.py` overrides the requested evaluation type to `NL_ASSERTIONS`). No test suite. Not built on Inspect.

Against our three requirements:

- **30+ step runs:** the step loop allows it (`max_steps` default 200), but tasks are short single-goal requests that resolve in a few turns. Long horizons would need new tasks and a user simulator that keeps feeding work, which is the bulk of the effort anyway.
- **Mid-trajectory injection:** not supported. Would need a scheduled DB mutation or scheduled user utterance hook in the orchestrator, plus per-task schedules.
- **Swappable compaction:** easy hook point (`LLMAgent.generate_next_message` appends to a plain message list), but no notion of a memory policy, pinned region, or per-step snapshot.
- **Ground truth:** LLM-judge only. Our design needs deterministic markers. Its tasks have no planted marker tokens or resolve-vs-preserve labels.

**Recommendation: build the minimal synthetic document world for the pilot, on Inspect.** Extending KC-Bench would mean new tasks, a new simulator mode, an injection scheduler, a new scorer, and a new logging layer, while also carrying a user simulator we do not need (it adds a second stochastic LLM to every trajectory, which is a confound for survival analysis). That is more than a day and still leaves us outside Inspect. What we keep from KC-Bench: its personalAssistant "stale record vs current record" pattern as a template for `resolve_by_rule` contradictions, and its domains as a phase-2 transfer target and a basis for contacting the authors.

## 1. Model

**Primary: Qwen3.6-35B-A3B (MoE, ~3B active), 4-bit AWQ, served by vLLM in WSL2, non-thinking mode.**
**Fallback: Qwen/Qwen3-32B-AWQ (official Qwen AWQ build, dense), thinking disabled.**

Rationale:
- Both fit in 32 GB with room for a 64k-token KV cache (4-bit weights are roughly 17 to 20 GB).
- The MoE model decodes several times faster than the dense 32B, which matters because active probes multiply calls. Qwen3-32B-AWQ is the better-documented choice for tool calling on a 5090 under vLLM, so it is the fallback if the MoE build misbehaves.
- Non-thinking mode for the pilot: shorter outputs, cheaper runs, fewer parsing problems. Thinking on vs off is a phase-1 covariate (DERELAB found extended thinking reduces confirmation bias).

Caveats I want flagged before we commit:
- I found Qwen3.6-35B-A3B only in third-party 2026 guides, not in an official model card I read. Before downloading, I will confirm the exact Hugging Face repo ID, who published the AWQ build, its size, and its upload date, and report those to you. If no trustworthy AWQ build exists, we use the fallback.
- Blackwell (sm_120) support in vLLM has been patchy. Community reports say AWQ may need `--quantization awq_marlin` and attention may need `--attention-backend TRITON_ATTN`. Gate check before any harness work: a 20-call tool-calling test (valid JSON tool calls, correct argument names).
- Exact model ID, quantization, vLLM version, and torch version get recorded in NOTES.md and in every run log.

Serving config (initial): `--max-model-len 65536`, `--gpu-memory-utilization 0.90`, `--enable-prefix-caching`, `--enable-auto-tool-choice` with the Qwen tool parser, temperature 0.6, top_p 0.95, max output 1024 tokens per step.

### Proposed install commands (not run yet, awaiting your OK)

Inside WSL (vLLM pulls a large CUDA torch build, roughly 10 GB):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv ~/venvs/vllm --python 3.12 && source ~/venvs/vllm/bin/activate && uv pip install vllm --torch-backend=auto
```

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Model download (roughly 17 to 20 GB) only after I report the exact repo ID:

```bash
hf download <confirmed-repo-id> --local-dir ~/models/<name>
```

On Windows, the harness itself (no GPU needed; it calls the vLLM endpoint at localhost:8000 through WSL2 port forwarding):

```bash
uv init --python 3.12 && uv add inspect-ai openai pyyaml numpy pandas lifelines matplotlib pytest
```

Ollama is not needed. If vLLM cannot be made to work on sm_120 within a few hours, Ollama with a Q4 GGUF of the same model is the fallback, at the cost of less control over sampling and batching.

## 2. Task family: fictional policy memo

The agent is a policy analyst for a fictional regional government (example: the Velmora Basin Water Authority). Brief: produce a briefing memo answering **6 broad decision questions** (for example "Can the Halvorsen Reservoir meet projected 2028 demand?", "Which pipeline route should the Authority fund?"). The brief does not name the 40 tracked issues. (Revised 2026-11-08 at owner request: an explicit issue list made tracking unrealistically easy.)

Each of the 40 tracked issues (20 contradictions, 20 matched controls) is written to be **necessary evidence for one of the 6 questions**, so dropping it is a task-relevant loss and not a reasonable relevance judgment. Each question gets roughly 6 to 7 issues. Only the harness knows which issues carry a planted contradiction.

### Steps and tools

A step is one agent model call that produces one tool call. Run length is fixed at **40 steps**.

Tools: `read_doc(doc_id)`, `search(query)`, `write_note(text)`, `conclude_interim(text)`, `final_answer(memo)`.

Information arrives on a fixed schedule. This is how injection timing is controlled exactly. At scheduled steps the harness appends incoming documents ("inbox items") to the tool result the agent sees, whatever tool it called. Between deliveries, the agent reads background documents, searches, and takes notes.

- **Recoverable** items are also filed in the library and can be re-read with `read_doc` or found by `search`.
- **Non-recoverable** items are labeled as transient (for example a phone call readout) and are never filed. After compaction, they exist only in whatever the memory policy kept.

Forced events, so timing is deterministic and not left to the agent:
- **Step 10:** harness instructs the agent to call `conclude_interim` with its current position on every issue. This is the interim conclusion that `before_interim` and `after_interim` are measured against.
- **Step 40:** harness instructs the agent to call `final_answer`.

### Final answer format (enables deterministic scoring)

The memo answers each question in prose and ends with an **evidence table the agent fills itself**: one row per figure or claim it relied on, `question | claim | sources (doc ids) | status (settled / disputed / superseded)`. The agent chooses the rows. Final scoring is deterministic, from markers in the table plus the prose:
- `choose_a` / `choose_b`: the correct side's marker is present with status settled, and the wrong side's marker is absent or marked superseded.
- `preserve_both`: both markers present, both sources cited, status disputed.
- `omitted`: neither marker appears in the memo. This is a new outcome that the list-free design allows. It is scored as incorrect handling, and the per-step labels plus active probes show whether the issue was still held in memory at step 40 (know-but-don't-output) or lost earlier.

Rows that cannot be matched to any planted marker are kept and counted (an extraction-volume statistic) but not scored.

## 3. World generation and the 20 planted contradictions

`worlds/generator.py` is **template-based and fully deterministic given a seed**. No LLM is involved in writing planted claims, so ground truth is exact. Each world seed draws fictional entity names, numbers, dates, source names, and the schedule.

Per world:
- 40 issues: 20 contradiction issues and 20 control issues.
- About 30 background library documents (300 to 600 tokens each), carrying side A of most issues plus distractor content.
- About 25 scheduled inbox items carrying side B, control second mentions, and distractors.
- A total full-context budget of about 45k tokens by step 40, safely under 64k.

### Contradiction allocation (20 per world)

The 9 cells (magnitude x kind) get 2 contradictions each = 18, plus 2 extras (direct x must_preserve, numeric x resolve_by_rule) = 20.

Timing: side A always arrives in steps 1 to 6 (library doc pointed to by the brief, or early inbox). Side B arrival is split across three windows:
- `before_interim`: side B at steps 3 to 9 (7 contradictions)
- `after_interim, before_compaction`: steps 11 to 14 (6)
- `after_interim, after_compaction`: steps 16 to 26 (7)

Each cell's two contradictions are placed in different windows, and the window assignment rotates across world seeds so every cell sees every window across the 5 seeds.

Interim alignment (needed for H3): for `after_interim` contradictions, side A is the side the agent would have concluded on at step 10 (it is the only side available then). We record which side the interim conclusion actually took, from the `conclude_interim` text by marker, so the anchoring index uses observed rather than assumed alignment.

Recoverable: **I propose all 20 non-recoverable in the pilot**, because recoverable evidence lets the agent re-read and masks loss, and with only 20 per world, splitting this factor leaves cells too thin. Recoverable vs non-recoverable becomes a balanced factor in phase 1. (This is one of the questions at the end.)

### How each magnitude is built

- `direct`: side A "The pipeline will route through Grath Valley" vs side B "The pipeline will not route through Grath Valley; the Orren Pass route was selected." Markers: route names.
- `numeric`: side A "storage forecast 41.7%" vs side B "storage forecast 38.2%". Markers: the two numbers (unique within the world).
- `qualifier`: side A "is expected to exceed 120 ML/day" vs side B "will exceed 120 ML/day" (hedge removed), or the reverse. Markers: the hedge phrase plus a shared anchor number. Paraphrase can defeat substring matching here, so passive probing is weakest for qualifier contradictions; the active probe covers this, and I will report qualifier passive-probe recall separately.

### How each kind signals correct handling

- `resolve_by_rule`: the brief states the rules (newer dated record supersedes older; an official register supersedes press). Side B carries a later date or the register source.
- `resolve_by_credibility`: one side comes from a clearly authoritative source (Authority audit office) and the other from a low-credibility source (anonymous blog, lobbying group).
- `must_preserve`: two equally credible sources (two independent consultancies) with no rule to break the tie. The brief says genuinely disputed points must be reported as disputed, with attribution.

### Matched non-contradictory controls (required by CLAUDE.md section 8)

Each of the 20 control issues is matched one-to-one with a contradiction: the same magnitude-type surface form (number, route name, or hedged statement), first mention in the same step window as that contradiction's side A, and a second **consistent** mention at that contradiction's side-B step, with the same recoverability. Each mention has its own marker (for example "41.7%" and "Bryce Reservoir"). Survival for a control means the fact remains present and correct. Contradiction survival minus control survival, at the same steps and under the same policy, is the contradiction-specific effect. Without it, the result is generic forgetting.

## 4. Memory policies (3 for the pilot)

All share one interface (`MemoryPolicy.observe(step)`, `compact(step) -> context`, `pinned()`).

| Policy | Behavior |
|---|---|
| `full_context` | No compaction. Full message history every step. Upper bound. |
| `llm_summary` | At compaction steps, the same model summarizes everything except the system prompt, the brief, and the last 2 steps into a summary of at most 1,000 tokens, which then replaces that history. The summarizer prompt is generic ("preserve information needed to finish the task") and is not told about contradictions. Agent notes are part of the history, so they get summarized too. |
| `oracle_pin` | Identical to `llm_summary`, plus a pinned block, excluded from compaction and re-injected after every compaction, holding both sides of every ground-truth contradiction whose side B has arrived. Each side is quoted verbatim with its source and date. The block lists them as "claims on the same issue from different sources" and does **not** say which is correct. Ceiling for perfect detection plus retention. Controls are not pinned, so under oracle_pin the control comparison only shows what pinning protects. |

The 1,000-token summary budget is deliberately tight relative to 40 issues. It is the main "aggressiveness" knob if everything survives.

## 5. Compaction steps

**Compaction fires at steps 15 and 28** (after the agent's step-15 and step-28 actions, before step 16 and step 29). Each compaction is logged with its input and output. This gives:
- 13 steps of exposure after compaction 1 before compaction 2,
- some side-B arrivals before compaction 1 (survive two boundaries) and some between the boundaries (survive one),
- 12 steps after compaction 2 before the final answer.

## 6. Probe schedule

- **Passive probe, every step, all 40 issues.** Deterministic scan of the exact context the model saw at that step (messages + summary + pinned block + notes) for each side's markers and source ids. Writes one state label per contradiction per step: `undetected`, `held_both`, `collapsed_to_a`, `collapsed_to_b`, `resolved_correct`, `resolved_wrong`, `lost`. The label rules:
  - `held_both`: both markers present **and** a conflict link (both markers within the same note, summary paragraph, or pinned entry, or an explicit conflict word like "conflict", "discrepancy", "contradicts", "disputed" within a fixed window).
  - `resolved_*`: only one side remains, and agent-authored text in the window marks it as the chosen or superseding side.
  - `collapsed_*`: only one side remains, and there is no resolution language.
  - `lost`: neither side present.
  - `undetected` covers the steps where both sides are present in raw tool output but no agent-authored text links them.
  The rules are heuristic. Unit tests pin down their behavior on handwritten fixtures, and the active probe checks them.
- **Active probe, every 5 steps (steps 5, 10, ..., 40), contradiction issues whose side B has arrived plus their matched controls.** Fork the exact context, ask "What is your current understanding of <issue>, and on what sources is it based?", record the answer, and discard the fork. One fork per issue; forks run in parallel against vLLM. That is at most 40 probes per probe point, 8 probe points, so at most 320 probe calls per run.
- **Active-probe labeling:** marker matching first (deterministic). Only answers where marker matching is ambiguous go to an LLM judge. The pilot judge is the same local model with a fixed rubric prompt. I will hand-label 100 pilot probe answers and report agreement. The full 200-sample validation is phase 1, with a judge model different from the agent.

## 7. Seeds and conditions

- 5 seeds. Each seed is a (world_seed, agent_seed) pair. **The same world is used across all three policies for a given seed**, so policy comparisons are paired.
- 3 policies x 5 seeds = **15 runs**, each with 20 contradictions + 20 controls. Per policy: 100 contradiction trajectories and 100 control trajectories.
- Sampling: temperature 0.6, top_p 0.95, fixed seed passed to vLLM. vLLM is not bit-exact under batching; I will log this and not claim exact replay.

Statistical honesty: 100 contradiction trajectories per policy is enough to see a large survival gap between `full_context` and `llm_summary`, and maybe a magnitude effect. It is not enough for 9-cell or timing-by-magnitude interactions. Those are phase 1. The pilot results note will report Kaplan-Meier curves with CIs and say plainly which comparisons are underpowered.

## 8. Estimated runtime

Assumptions (to be replaced with measured numbers after the smoke test): MoE model at roughly 100 to 150 tok/s single-stream decode, with prefix caching making prefill cheap; about 300 output tokens per step.

| Component | Per run |
|---|---|
| 40 agent steps at ~3 to 5 s | 2 to 3.5 min |
| 2 compactions (~1,000 output tokens each) | ~0.5 min |
| 8 active-probe points, up to 40 parallel forks at ~200 tokens each | 2 to 4 min |
| **Total per run** | **~5 to 8 min** |

15 runs sequentially: **~1.5 to 2 hours**. Running 3 at once on vLLM: **~45 to 60 min**. Fallback dense Qwen3-32B-AWQ: roughly 2.5 to 3x longer (**~4 to 6 hours** sequential). Judge calls for ambiguous probes and hand labeling (~1 to 2 hours of your time for 100 labels) are extra. One-time setup (vLLM install, model download, tool-calling gate check) is likely 1 to 3 hours, more if sm_120 needs a source build.

API spend for the pilot: none. Frontier spot checks are phase 1.

## 9. Files to create, in build order

Each step ends with passing tests before the next starts.

1. `pyproject.toml` (uv, Python 3.12), `.gitignore` (runs/, .env, *.safetensors, model dirs, __pycache__), `README.md`, `NOTES.md`, `git init`.
2. `worlds/schema.py`: `Claim`, `PlantedContradiction`, `ControlFact`, `Document`, `World`, `Schedule` dataclasses (from CLAUDE.md section 4, plus controls).
3. `worlds/templates/`: sentence templates per magnitude and kind, entity and source name banks, distractor paragraphs.
4. `worlds/generator.py`: `generate_world(seed, config) -> World`.
5. `tests/test_generator.py`: determinism (same seed gives the same world), allocation (20 contradictions covering all 9 cells, 3 timing windows, rotation across seeds), marker uniqueness within a world, every control matched to a contradiction, side A always before side B, no marker leakage into the brief, token budget under 45k.
6. `probes/passive.py`: marker scan plus state-label rules.
7. `tests/test_passive_probe.py`: handwritten context fixtures for every state label, qualifier paraphrase cases (documenting known misses), control-fact survival.
8. `agent/memory/base.py`, `full_context.py`, `llm_summary.py`, `oracle_pin.py`, plus `tests/test_memory.py` (compaction fires at exactly 15 and 28; pinned block survives; brief and last 2 steps are kept; uses a fake model, no GPU).
9. `agent/tools.py`: the five tools, the library, and inbox delivery with the recoverable flag. `tests/test_tools.py`.
10. `agent/react_agent.py`: Inspect agent loop with fixed 40 steps, forced step-10 interim, forced step-40 final answer, memory policy hook, and per-step JSONL snapshot (full context, notes, summary, pinned block, tool call, output).
11. `probes/active.py`: forked probes (forks never written back), marker-first labeling, judge fallback.
12. `scoring/per_step.py`, `scoring/final.py`: state-label time series and findings-table parsing and scoring.
13. `tasks/policy_memo.py`: Inspect task wiring a world, a policy, and a seed.
14. `configs/pilot.yaml`, `configs/smoke.yaml`: model, endpoint, policies, seeds, compaction steps, probe interval, budgets.
15. **Smoke test:** 1 world, 1 seed, `llm_summary` (it exercises compaction), then `full_context`. Review the transcript by hand before any batch.
16. `analysis/survival.py`: Kaplan-Meier curves by policy and magnitude, hazard at compaction vs non-compaction steps, contradiction-minus-control gap.
17. Pilot batch (15 runs), then `analysis/pilot_results.md`.

## 10. Owner decisions (2026-11-08)

1. **Model:** approved QuantTrio/Qwen3.6-35B-A3B-AWQ (verified: third-party 4-bit AWQ of the official model, uploaded 2026-04-17, 25.5 GB with vision tower). Fallback Qwen/Qwen3-32B-AWQ (19.3 GB). The 25.5 GB size leaves little KV headroom; context length is confirmed at first load.
2. **Recoverability:** all non-recoverable in the pilot. Recoverable vs non-recoverable is a required measured factor in phase 1, since most real contradictions are recoverable.
3. **Installs:** approved and done (vLLM 0.33.0, torch 2.14.0+cu132 in WSL).
4. **Issue list:** dropped. Replaced by 6 broad questions plus an agent-filled evidence table (section 2).
5. **API budget:** $500 hard cap for the whole project. Pilot is $0. A measured cost table is required before any frontier run.
