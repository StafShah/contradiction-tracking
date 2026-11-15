# NOTES

Running log of decisions and results. Newest entries at the bottom.

## 2026-11-08

- Environment check. Windows: RTX 5090 32,607 MiB, driver 610.88; Python 3.14.7; uv 0.12.3; git 2.55.0. WSL2 Ubuntu-24.04 with GPU visible, Python 3.12.3, no uv, no torch, no vLLM. No Ollama anywhere. Nothing installed yet.
- KC-Bench cloned to ../KC-Bench (commit adf93de, 2026-07-28). Assessment: a tau2-bench fork with static t=0 conflicts, an LLM user simulator, LLM-judge-only scoring, no tests, not Inspect-based. Decision (pending owner approval): build a minimal synthetic document world on Inspect for the pilot. Reuse KC-Bench's stale-vs-current-record pattern for resolve_by_rule. Revisit KC-Bench as a phase-2 transfer target.
- Proposed model: Qwen3.6-35B-A3B 4-bit AWQ (primary, repo ID to be verified) or Qwen/Qwen3-32B-AWQ (fallback), via vLLM in WSL2, non-thinking mode. Reason: fits 32 GB with 64k KV cache; the MoE model is faster for probe-heavy runs.
- Pilot design drafted in PILOT_PLAN.md: 40 steps, interim conclusion forced at step 10, compaction at steps 15 and 28, 20 contradictions + 20 matched controls per world, 3 policies x 5 paired seeds. Awaiting owner approval before any harness code.
- Owner approved the plan. vLLM 0.33.0 + torch 2.14.0+cu132 installed in WSL at ~/venvs/vllm; torch sees the RTX 5090 as capability (12, 0). Model load not yet tested.
- Model: owner approved QuantTrio/Qwen3.6-35B-A3B-AWQ (third-party 4-bit AWQ of Qwen/Qwen3.6-35B-A3B, uploaded 2026-04-17, 25.5 GB including vision tower). Fallback: Qwen/Qwen3-32B-AWQ (official, 19.3 GB). Download started to ~/models/.
- API budget: hard cap of $500 total for the project. Pilot uses $0. Before any frontier run, measure tokens per run from the smoke test and present a cost table for approval. Frontier runs will likely need a reduced probe schedule.
- Recoverability: pilot stays all non-recoverable (clean signal for memory loss). Owner notes most real contradictions are recoverable, so recoverable vs non-recoverable must be a measured factor in phase 1, not an afterthought.
- Known fix pending: uv created the venv on Python 3.14.7; pin to 3.12 as planned.
- Python pinned to 3.12 (uv had defaulted to 3.14). inspect-ai 0.3.288.
- Model downloaded to ~/models/Qwen3.6-35B-A3B-AWQ (24 GB on disk; index sha256 a3bfee6f...). Architecture Qwen3_5MoeForConditionalGeneration: 40 layers, 10 full-attention (rest linear attention), 2 KV heads x 256 dim, so KV is about 20 KB/token and a 64k context needs about 1.3 GB. Weights take 21.3 GiB in vLLM.
- vLLM startup fix: FlashInfer sampler JIT needs nvcc, which WSL lacks. Set VLLM_USE_FLASHINFER_SAMPLER=0 (PyTorch sampler) instead of installing a CUDA toolkit. Serve script: scripts/serve_vllm.sh (--language-model-only, 64k context, prefix caching, qwen3_coder tool parser, qwen3 reasoning parser, thinking disabled per request).
- dtype: AWQ defaults to float16 while the checkpoint is bf16. The first successful server came up in fp16 (a script edit for bf16 failed to apply). Script now defaults to --dtype bfloat16; gate check includes an arithmetic sanity prompt.
- World generator: template-based, deterministic. Actual full-world size is about 14k tokens (112 docs), below the 45k planned; context pressure comes mostly from agent outputs. Note for the Adjust branch.
- Qualifier detection: hedged side is marked by an invented condition token (Claim.condition_marker) plus hedge words in the anchor's clause; certain side is a bare anchor. Dropping a hedge therefore reads as collapse to the certain claim, by design.
- Tests: 144 passing (generator, passive probe, memory policies, tools, final scoring).
- Decision: no explicit issue list. Brief gives 6 broad decision questions; each of the 40 tracked issues is necessary evidence for one question. Memo ends with an agent-filled evidence table (question, claim, sources, status). New final outcome `omitted` (neither marker in memo). Why: a named list makes tracking unrealistically easy; tying issues to questions keeps every planted fact task-relevant so a drop is a loss, not a relevance judgment.
- vLLM second failure: max_num_seqs 256 exceeded linear-attention (Mamba) cache blocks (248). Fix: --max-num-seqs 64. Server now up in bf16 with 5.9 GiB KV, 265,888 tokens, about 4 concurrent 64k contexts.
- Gate check (scripts/gate_tool_calling.py): first run 18/20; both failures were forced final_answer calls truncated at max_tokens 1024 (empty arguments). Added interim_max_tokens 2048 and final_max_tokens 4096. Rerun: 20/20 valid tool calls, arithmetic sanity correct. GATE PASS.
- Inspect could not import the repo packages; made the project an editable hatchling package.

### Smoke test 1 (world 0, seed 0, llm_summary), 2026-11-08
- Ran end to end: 40 steps, compactions at 15 and 28, active probes at every 5th step, 88 s wall time, 1.78M tokens total (agent 275k in, probes 1.45M in, summarizer 20k in). Runtime is far below the 5 to 8 min/run estimate.
- Behavior: the agent made zero write_note calls in 40 steps. It searched and read library docs and left transient inbox items unrecorded. The summarizer nonetheless preserved many transient items with doc ids, editions, and hedges.
- Compaction 1: 49.6k chars in, 3.7k chars out. Passive probe: 5 held_both and 8 undetected before compaction; after it, 7 lost.
- Final (after probe fixes, via analysis/rescore.py): 6/20 correct, 2 wrong_choice, 1 wrongly_resolved, 11 omitted; control recall 10/20. Omission dominates, for contradictions and controls alike, so at n=1 this looks mostly like generic loss plus memo selectivity. No contradiction-specific claim is possible yet.
- Qualitative errors worth tracking: C07 and C01 kept the lower edition number as "later", consistent with arrival order overriding the edition/date rule. C04 (must_preserve) collapsed a hedge into "confirmed... settled".
- Design observation: after_compaction contradictions often start as collapsed_to_b because side A was already compacted away when B arrived. Analysis must separate "never co-present" from "detected then lost".
- Harness fixes from the smoke test: early final_answer now rejected before step 40 (fixed horizon; the agent tried at step 39 and was truncated); normal-step max_tokens 2048; steps.jsonl now logs the pre-action context (what the model and passive probe saw), not the post-action one.
- Probe fixes from the smoke test (each with a regression test): qualifier hedge words now attach to a specific decimal number (leading hedges to the next number, trailing conditions to the previous), fixing a leak across comma-separated list items; a comma-split attempt broke a table cell and was replaced; added "the current reliable figure is X" as a choice cue; widened don't-know detection (needs_judge fell from 89 to 18 of 234 probe answers).

### Smoke test 2 (world 0, seed 0, all three policies), 2026-11-08
- First full_context run scored 0/20: a harness artifact. Its memo exceeded final_max_tokens 4096, the truncated forced call had empty arguments, and the scorer fell back to stray assistant text. Fixes: final_max_tokens 8192; one retry at 16384 if a forced call is truncated (logged in result.forced_retries); a run with no memo is now marked valid=False and scored as an empty memo, never as stray text. Rerun of full_context: valid, memo 4,022 tokens, no retry.
- Final-memo results (n = 1 world, 1 seed each; rescored with current probe code):
  - full_context: 8/20 correct, 10 omitted, 1 wrongly_preserved, 1 wrongly_resolved; controls 10/20.
  - llm_summary: 6/20 correct, 8 omitted, 2 wrong_choice, 3 wrongly_resolved, 1 wrongly_preserved; controls 9/20. (Smoke test 1 on the same world: also 6/20.)
  - oracle_pin: 19/20 correct, 1 omitted; controls 10/20.
- Honest reading: with n = 1 these are not results. Two things already matter for design:
  1. Even full_context omits about half of both contradictions and controls from the memo. With 6 broad questions the memo cannot cover all 40 issues, so final-memo accuracy mostly measures memo selectivity. The per-step passive and active probes should be the primary survival measures, with final scoring as a secondary outcome conditioned on the item being mentioned.
  2. oracle_pin's near-perfect score is partly a salience cue: the pinned block tells the agent which issues matter. Its control recall (10/20) stays at the other policies' level, consistent with pinning helping only what it pins. The oracle_pin vs naive_pin vs conflict_ledger comparison in phase 1 needs a salience-matched control (for example pinning both sides plus an equal number of control facts).
- Cost per run (input tokens): full_context about 4.9M (probes about 4.2M), llm_summary about 1.6M, oracle_pin about 2.3M. Probes dominate. Wall time 70 to 95 s per run, so the 15-run pilot is about 25 min.
- Passive labels under oracle_pin count the pinned block as agent state, so held_both there is partly tautological. Report oracle_pin passive labels with and without the pinned segment.

### Pilot batch, 2026-11-08 (runs/20261108_152817_pilot)
- 15 runs (3 policies x 5 paired seeds), all valid, 14 min 42 s, 44.9M tokens, $0. Ran 3 episodes in parallel (each run 2 to 4 min under contention).
- Analysis provenance: the survival event definitions (analysis/survival.py docstring), the "failed at arrival" split, and the absolute-step view were written before the pilot finished, informed only by smoke-test data. The two-fact fairness baseline and the per-run table were added AFTER seeing pilot results. The fairness check is post hoc and is labeled that way here.
- Decision: GO for phase 1, with design changes. Full write-up in analysis/pilot_results.md.
- Headline: under llm_summary, contradictions fail more often than matched controls in 5/5 seeds; S(20) 0.44 vs 0.71 (log-rank p = 0.0001); both kinds below fair baselines (resolve 0.53 vs 0.71; must_preserve 0.29 vs a two-fact baseline of 0.50). Per compaction, contradictions are lost at about 1.9x the control rate.
- H1 suggestive (qualifier S(20) 0.25 vs direct 0.46, numeric 0.60; CIs overlap). H3 not supported (collapse not toward the interim side). The passive boundary-hazard spike is built into the non-recoverable design; report only the contrast.
- Unexpected: oracle_pin lost more control facts than llm_summary (S(20) 0.56 vs 0.71, 4/5 seeds worse, 1 tie). Mechanism unknown.
- Localization: llm_summary failures are upstream (probe already wrong, memo omits: 35/100); oracle_pin failures are at synthesis (probe correct, memo omits: 34/100).
- Variance: same world and seed gave oracle_pin 0.95 (smoke) vs 0.55 (batch) final accuracy; vLLM batching is nondeterministic. Note-taking is bimodal per run (0 or about 20).

## 2026-11-14

- Repo put under version control. Convention documented in README.md: `type(scope): summary` commits, one `exp(...)` commit per batch, annotated tags `exp/<date>-<name>` for each completed batch, `phase<N>/<topic>` branches for large changes. Commit before every batch so `meta.json` records the code version.
- Provenance caveat: the pilot batch ran before the first commit, so its `meta.json` files have `git_commit: null`. The harness code committed here is the code that produced the pilot. The analysis code also includes the post hoc fairness baseline and per-run table, added after the batch (see the pilot entry above).
- vLLM server is stopped; restart with `bash scripts/serve_vllm.sh` in WSL before new runs.
