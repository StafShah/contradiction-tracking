"""Fixed ReAct-style agent loop with a swappable memory policy.

One step = one agent model call = at most one executed tool call.
Per step t:
  1. policy.on_step(t); the step message (inbox items, step counter, any forced
     instruction) is added to context.
  2. Passive probe on the exact context. Active probe (discarded forks) every K steps.
  3. Model call. The first tool call is executed; extra calls get an error result.
  4. If t is a compaction step, policy.compact(t) runs after the action.
Everything is written to runs/<run_id>/ as JSON or JSONL.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
)
from inspect_ai.tool import ToolFunction

from agent.memory.base import POLICIES, MemoryPolicy, message_text
from agent.tools import TOOLS, DocEnvironment
from probes.active import run_active_probe, to_dicts as probe_dicts
from probes.passive import PassiveProbe
from scoring.final import score_memo, summarize, to_dicts as score_dicts
from worlds.schema import World

SYSTEM_PREAMBLE = "You are a careful policy analyst working through a multi-step research assignment using tools. Make exactly one tool call per step."

SUMMARIZER_SYSTEM = "You compress the working history of an analyst agent so that it can continue its assignment after the history is removed."
SUMMARIZER_USER = (
    "Below is the analyst's working history so far. Write a summary that lets the analyst continue the assignment. "
    "Preserve the information needed to finish the task. Keep the summary under {budget} tokens. "
    "Write the summary only.\n\n<history>\n{history}\n</history>"
)


@dataclass
class AgentConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 2048
    interim_max_tokens: int = 2048  # conclude_interim covers six questions
    final_max_tokens: int = 8192  # memo plus evidence table; 4096 truncated a full_context memo in smoke test 2
    forced_retry_max_tokens: int = 16384  # one retry if a forced call is truncated
    seed: int = 0
    enable_thinking: bool = False
    probe_interval: int = 5
    active_probe: bool = True
    probe_max_tokens: int = 200
    probe_concurrency: int = 16
    summary_budget_tokens: int = 1000
    keep_last_steps: int = 2

    def generate_config(self, max_tokens: int | None = None, seed_offset: int = 0) -> GenerateConfig:
        return GenerateConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens or self.max_tokens,
            seed=self.seed + seed_offset,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        )


@dataclass
class EpisodeResult:
    run_id: str
    policy: str
    world_seed: int
    agent_seed: int
    final_memo: str | None
    final_summary: dict
    n_steps_run: int
    tokens_in: int = 0
    tokens_out: int = 0
    probe_tokens_in: int = 0
    probe_tokens_out: int = 0
    summary_tokens_in: int = 0
    summary_tokens_out: int = 0
    wall_seconds: float = 0.0
    tool_errors: int = 0
    no_tool_steps: int = 0
    reads_failed: int = 0
    run_dir: str = ""
    valid: bool = True  # False if no memo was submitted (a harness failure, not a model outcome)
    forced_retries: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def make_summarizer(model: Model, cfg: AgentConfig, counters: dict):
    async def summarize_fn(messages: list[ChatMessage], budget: int) -> str:
        history = "\n\n".join(f"[{m.role}] {message_text(m)}" for m in messages)
        out = await model.generate(
            input=[ChatMessageSystem(content=SUMMARIZER_SYSTEM),
                   ChatMessageUser(content=SUMMARIZER_USER.format(budget=budget, history=history))],
            config=cfg.generate_config(max_tokens=int(budget * 1.2), seed_offset=10_000),
        )
        if out.usage:
            counters["summary_in"] += out.usage.input_tokens
            counters["summary_out"] += out.usage.output_tokens
        return out.completion.strip()

    return summarize_fn


def step_message(world: World, env: DocEnvironment, t: int) -> str:
    cfg = world.config
    lines = [f"[Step {t} of {cfg.n_steps}. Steps remaining after this one: {cfg.n_steps - t}.]"]
    items = env.inbox(t)
    if items:
        lines.append(f"Inbox: {len(items)} new item(s).\n")
        lines += [d.render() + "\n" for d in items]
    else:
        lines.append("Inbox: no new items.")
    if t == cfg.interim_step:
        lines.append("Required now: call conclude_interim with your current position on each question (Q1 to Q6) and the key evidence for it.")
    if t == cfg.final_step:
        lines.append("Required now: this is the final step. Call final_answer with the complete memo in the required format, including the evidence table.")
    return "\n".join(lines)


def _jsonl(path: Path, obj) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


async def run_episode(world: World, policy_name: str, model: Model, cfg: AgentConfig, run_dir: Path, run_id: str) -> EpisodeResult:
    t0 = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    for fn in ("steps.jsonl", "passive.jsonl", "active.jsonl"):
        (run_dir / fn).unlink(missing_ok=True)
    (run_dir / "world.json").write_text(json.dumps(world.to_dict(), indent=1), encoding="utf-8")

    counters = {"summary_in": 0, "summary_out": 0}
    system_prompt = SYSTEM_PREAMBLE + "\n\n" + world.brief
    policy: MemoryPolicy = POLICIES[policy_name](
        world, system_prompt, summarizer=make_summarizer(model, cfg, counters),
        summary_budget_tokens=cfg.summary_budget_tokens, keep_last_steps=cfg.keep_last_steps,
    )
    env = DocEnvironment(world)
    passive = PassiveProbe(world)
    res = EpisodeResult(run_id, policy_name, world.seed, cfg.seed, None, {}, 0, run_dir=str(run_dir))
    last_text = ""

    for t in range(1, world.config.n_steps + 1):
        policy.on_step(t)
        policy.add(ChatMessageUser(content=step_message(world, env, t)), "env", t)

        # -- passive probe on the exact context the model is about to see
        segs = policy.segments()
        pr = passive.step(t, segs)
        _jsonl(run_dir / "passive.jsonl", {"step": t, "contradictions": {k: asdict(v) for k, v in pr.contradictions.items()},
                                           "controls": {k: asdict(v) for k, v in pr.controls.items()}})

        context = policy.context()

        # -- active probe in discarded forks
        if cfg.active_probe and t % cfg.probe_interval == 0:
            answers = await run_active_probe(model, context, world, t, TOOLS,
                                             cfg.generate_config(max_tokens=cfg.probe_max_tokens, seed_offset=20_000 + t),
                                             concurrency=cfg.probe_concurrency)
            for a in answers:
                res.probe_tokens_in += a.input_tokens or 0
                res.probe_tokens_out += a.output_tokens or 0
            for d in probe_dicts(answers):
                _jsonl(run_dir / "active.jsonl", d)

        # -- agent action
        forced, max_tok = None, cfg.max_tokens
        if t == world.config.interim_step:
            forced, max_tok = ToolFunction(name="conclude_interim"), cfg.interim_max_tokens
        elif t == world.config.final_step:
            forced, max_tok = ToolFunction(name="final_answer"), cfg.final_max_tokens
        ts = time.time()
        out = await model.generate(input=context, tools=TOOLS, tool_choice=forced or "auto",
                                   config=cfg.generate_config(max_tokens=max_tok, seed_offset=t))
        latency = time.time() - ts
        if out.usage:
            res.tokens_in += out.usage.input_tokens
            res.tokens_out += out.usage.output_tokens
        # A forced call cut off by the token cap arrives with empty arguments. Retry once
        # with a larger cap; the truncated attempt never enters the trajectory.
        truncated = out.stop_reason in ("max_tokens", "model_length") and not any((tc.arguments or {}) for tc in (out.message.tool_calls or []))
        if forced is not None and truncated:
            res.forced_retries.append(t)
            out = await model.generate(input=context, tools=TOOLS, tool_choice=forced,
                                       config=cfg.generate_config(max_tokens=cfg.forced_retry_max_tokens, seed_offset=t))
            latency = time.time() - ts
            if out.usage:
                res.tokens_in += out.usage.input_tokens
                res.tokens_out += out.usage.output_tokens
        msg: ChatMessageAssistant = out.message
        policy.add(msg, "agent", t)
        last_text = msg.text or last_text

        tool_record = None
        done = False
        if msg.tool_calls:
            for i, tc in enumerate(msg.tool_calls):
                if i == 0:
                    r = env.execute(t, tc.function, tc.arguments or {})
                    tool_record = {"name": tc.function, "arguments": tc.arguments, "result": r.text, "error": r.is_error}
                    res.tool_errors += int(r.is_error)
                    done = r.done
                    text = r.text
                else:
                    text = "Only one tool call is executed per step. This call was not executed."
                policy.add(ChatMessageTool(content=text, tool_call_id=tc.id, function=tc.function), "env", t)
        else:
            res.no_tool_steps += 1

        _jsonl(run_dir / "steps.jsonl", {
            "step": t, "policy": policy_name, "latency_s": round(latency, 3), "stop_reason": out.stop_reason,
            "usage": out.usage.model_dump() if out.usage else None,
            "assistant_text": msg.text, "tool": tool_record,
            # pre-action context: exactly what the model and the passive probe saw at this step
            "context": [{"kind": s.kind, "step": s.step, "text": s.text} for s in segs],
            "summary": policy.summary, "pinned": policy.pinned(),
        })
        res.n_steps_run = t
        if done:
            break

        if t in world.config.compaction_steps:
            await policy.compact(t)
            if policy.compactions:
                _jsonl(run_dir / "steps.jsonl", {"step": t, "event": "compaction", **asdict(policy.compactions[-1])})

    memo = env.final_memo
    if memo is None:
        res.valid = False
        res.notes.append("no memo submitted: run is INVALID (harness failure); scored as an empty memo")
        memo = ""
    res.final_memo = memo
    scores = score_memo(world, memo)
    res.final_summary = {**summarize(scores), "valid": res.valid}
    res.summary_tokens_in, res.summary_tokens_out = counters["summary_in"], counters["summary_out"]
    res.reads_failed = sum(1 for _, _, ok in env.reads if not ok)
    res.wall_seconds = round(time.time() - t0, 1)

    (run_dir / "final.json").write_text(json.dumps({
        "memo": memo, "scores": score_dicts(scores), "summary": res.final_summary,
        "interim": env.interim, "notes": env.notes, "early_final_attempts": env.early_final_attempts, "reads": env.reads, "searches": env.searches,
        "compactions": [asdict(c) for c in policy.compactions],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(asdict(res), indent=1, ensure_ascii=False), encoding="utf-8")
    return res
