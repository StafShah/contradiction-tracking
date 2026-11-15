# Contradiction Tracking in Long-Horizon LLM Agents

When an LLM agent meets a contradiction partway through a long task, does correct handling of it survive the rest of the task? This repo measures how long contradictions stay represented, whether they are resolved or preserved correctly, and which events (context compaction, an interim conclusion) drive failure. It then compares memory policies, including a conflict ledger, against pinning baselines.

Status: pilot complete (2026-11-08, decision GO with design changes). Results in `analysis/pilot_results.md`. See `PILOT_PLAN.md` for the design and `NOTES.md` for the dated decision log.

## Layout

```
worlds/      fictional document worlds with planted contradictions and matched controls (deterministic generator)
agent/       fixed ReAct-style agent loop, tools, and swappable memory policies
probes/      passive probe (deterministic marker scan every step) and active probe (discarded forks)
scoring/     final-memo scoring from planted markers
tasks/       Inspect task definition
configs/     experiment configs (smoke, pilot)
scripts/     vLLM serve script, tool-calling gate check
tests/       unit tests (run before anything expensive)
runs/        raw run logs (gitignored)
```

## Running

1. Serve the model inside WSL2: `bash scripts/serve_vllm.sh`
2. Gate check: `uv run python scripts/gate_tool_calling.py`
3. Tests: `uv run pytest -q`
4. Smoke test (1 world, 1 seed):

```bash
uv run inspect eval tasks/policy_memo.py -T config=configs/smoke.yaml --model openai-api/vllm/qwen3.6-35b-a3b-awq --log-dir runs/inspect_logs
```

Set `VLLM_BASE_URL=http://localhost:8000/v1` and `VLLM_API_KEY=local` first (see `.env.example`). Each episode writes `world.json`, `steps.jsonl` (full context per step), `passive.jsonl`, `active.jsonl`, `final.json`, `result.json`, and `meta.json` under `runs/<batch>/<sample>/`.

## Model

Pilot model: `QuantTrio/Qwen3.6-35B-A3B-AWQ` (4-bit AWQ of `Qwen/Qwen3.6-35B-A3B`), vLLM 0.33.0, bf16 activations, thinking disabled, on a single RTX 5090.

## Experiment milestones

Each completed experiment batch gets an annotated git tag. The tag message records the batch id, config, and headline numbers. Raw logs stay local under `runs/<batch>/` and are never committed.

| Tag | Date | What | Results |
|---|---|---|---|
| `exp/2026-11-08-pilot` | 2026-11-08 | Pilot: 1 model x 5 worlds x 3 policies (15 runs), batch `20261108_152817_pilot` | `analysis/pilot_results.md` |

## Commit and versioning conventions

**Commit messages** use `type(scope): summary`, imperative, under about 70 characters, with a body explaining why when it is not obvious.

| Type | Use for |
|---|---|
| `feat` | new harness capability (a policy, a probe, a tool, a world feature) |
| `fix` | bug fix; say whether it changes any previously reported number |
| `test` | tests only |
| `exp` | an experiment batch: its config, regenerated figures, results note, and NOTES.md entry, together in one commit |
| `analysis` | analysis code or write-ups that do not come from a new batch (rescoring, new statistics) |
| `docs` | plan, notes, README, CLAUDE.md |
| `chore` | dependencies, tooling, repo housekeeping |

Scopes match top-level folders: `worlds`, `agent`, `memory`, `probes`, `scoring`, `tasks`, `analysis`, `configs`, `scripts`.

**Experiment workflow:**

1. Commit all code and the config before launching a batch, so the batch's `meta.json` records a real `git_commit` (it records `null` on an uncommitted tree).
2. Run the batch into `runs/<YYYYMMDD_HHMMSS>_<name>/`.
3. Make one `exp(<name>): ...` commit containing the regenerated figures, the results note, and a dated NOTES.md entry.
4. Tag it: `git tag -a exp/<YYYY-MM-DD>-<name>`, with the batch id, config path, and headline numbers in the message. Add a row to the milestones table above.

**Branches:** `main` holds working, tested code. Larger changes (a new policy, a probe redesign, a new scenario) go on `phase<N>/<topic>` branches and merge to `main` when tests pass.

**Hard rules:** never commit keys, `.env`, raw run logs, or model weights. A `fix` that changes a reported number must say so in the commit body and in NOTES.md.
