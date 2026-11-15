"""Inspect task: fictional policy-memo research with planted contradictions.

Run (from the repo root, with the vLLM server up in WSL):
    uv run inspect eval tasks/policy_memo.py -T config=configs/smoke.yaml \
        --model openai-api/vllm/qwen3.6-35b-a3b-awq

One sample per (world seed, memory policy). The solver runs the full episode
(agent loop, probes, compactions) and writes everything under runs/.
Inspect's own log captures every model call as the transcript.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver

from agent.react_agent import AgentConfig, run_episode
from worlds.generator import generate_world
from worlds.schema import WorldConfig

REPO = Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["_path"] = str(p)
    return cfg


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _world_config(cfg: dict) -> WorldConfig:
    w = dict(cfg.get("world", {}))
    if "compaction_steps" in w:
        w["compaction_steps"] = tuple(w["compaction_steps"])
    if "windows" in w:
        w["windows"] = {k: tuple(v) for k, v in w["windows"].items()}
    return WorldConfig(**w)


@solver
def memo_agent(cfg: dict):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        md = state.metadata
        world = generate_world(md["world_seed"], _world_config(cfg))
        agent_cfg = AgentConfig(**{**cfg.get("agent", {}), "seed": md["agent_seed"]})
        model = get_model()
        run_dir = REPO / "runs" / md["batch_id"] / str(state.sample_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "meta.json").write_text(json.dumps({
            "sample_id": state.sample_id, "batch_id": md["batch_id"], "config_path": cfg["_path"],
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "model": str(model), "model_notes": cfg.get("model_notes"),
            "world_seed": md["world_seed"], "agent_seed": md["agent_seed"], "policy": md["policy"],
            "world_config": asdict(world.config), "agent_config": asdict(agent_cfg),
            "git_commit": _git_commit(), "started": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
        }, indent=1), encoding="utf-8")
        res = await run_episode(world, md["policy"], model, agent_cfg, run_dir, str(state.sample_id))
        state.output = ModelOutput.from_content(model=str(model), content=res.final_memo or "")
        state.metadata["result"] = asdict(res)
        state.completed = True
        return state

    return solve


@scorer(metrics=[mean()])
def memo_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        r = state.metadata.get("result", {})
        s = r.get("final_summary", {})
        return Score(value=s.get("contradiction_accuracy", 0.0), answer=None,
                     explanation=json.dumps(s), metadata={"final_summary": s, "run_dir": r.get("run_dir")})

    return score


@task
def policy_memo(config: str = "configs/smoke.yaml"):
    cfg = load_config(config)
    batch_id = f"{datetime.now():%Y%m%d_%H%M%S}_{Path(cfg['_path']).stem}"
    samples = []
    for ws in cfg["world_seeds"]:
        for pol in cfg["policies"]:
            agent_seed = cfg.get("agent_seed_offset", 0) + ws
            samples.append(Sample(
                id=f"{pol}_w{ws}_a{agent_seed}", input="policy memo episode",
                metadata={"world_seed": ws, "agent_seed": agent_seed, "policy": pol, "batch_id": batch_id},
            ))
    return Task(dataset=MemoryDataset(samples), solver=memo_agent(cfg), scorer=memo_scorer())
