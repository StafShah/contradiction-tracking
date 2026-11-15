"""Tool-calling gate check (PILOT_PLAN.md section 1), run before any episode.

20 calls against the served model using the real agent tools and a real world
brief. Pass criteria: every call returns exactly one tool call to a known tool
with the required argument names, forced tool_choice is honored, and a fixed
arithmetic sanity prompt is answered correctly (catches fp16 numeric problems).

    uv run python scripts/gate_tool_calling.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.tool import ToolFunction

from agent.react_agent import SYSTEM_PREAMBLE, AgentConfig, step_message
from agent.tools import TOOLS, DocEnvironment
from worlds.generator import generate_world

MODEL = os.environ.get("GATE_MODEL", "openai-api/vllm/qwen3.6-35b-a3b-awq")
REQUIRED = {t.name: set(t.parameters.required) for t in TOOLS}


async def main() -> int:
    model = get_model(MODEL, base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
                      api_key=os.environ.get("VLLM_API_KEY", "local"))
    world = generate_world(0)
    env = DocEnvironment(world)
    cfg = AgentConfig()
    system = ChatMessageSystem(content=SYSTEM_PREAMBLE + "\n\n" + world.brief)

    cases = []
    for i in range(16):
        step = 1 + (i % 6)
        cases.append(("auto", [system, ChatMessageUser(content=step_message(world, env, step))], None))
    cases.append(("forced_interim", [system, ChatMessageUser(content=step_message(world, env, 10))], "conclude_interim"))
    cases.append(("forced_interim", [system, ChatMessageUser(content=step_message(world, env, 10))], "conclude_interim"))
    cases.append(("forced_final", [system, ChatMessageUser(content=step_message(world, env, 40))], "final_answer"))
    cases.append(("forced_final", [system, ChatMessageUser(content=step_message(world, env, 40))], "final_answer"))

    async def one(i, kind, msgs, forced):
        t = time.time()
        max_tok = {"conclude_interim": cfg.interim_max_tokens, "final_answer": cfg.final_max_tokens}.get(forced, cfg.max_tokens)
        out = await model.generate(input=msgs, tools=TOOLS,
                                   tool_choice=ToolFunction(name=forced) if forced else "auto",
                                   config=cfg.generate_config(max_tokens=max_tok, seed_offset=i))
        calls = out.message.tool_calls or []
        problems = []
        if len(calls) != 1:
            problems.append(f"{len(calls)} tool calls")
        for tc in calls:
            if tc.function not in REQUIRED:
                problems.append(f"unknown tool {tc.function}")
            elif not REQUIRED[tc.function] <= set((tc.arguments or {}).keys()):
                problems.append(f"missing args for {tc.function}: {tc.arguments}")
            if tc.parse_error:
                problems.append(f"parse error: {tc.parse_error}")
        if forced and calls and calls[0].function != forced:
            problems.append(f"forced {forced} but got {calls[0].function}")
        return {
            "i": i, "kind": kind, "ok": not problems, "problems": problems,
            "calls": [(tc.function, {k: str(v)[:80] for k, v in (tc.arguments or {}).items()}) for tc in calls],
            "text": (out.message.text or "")[:120], "latency_s": round(time.time() - t, 2),
            "in": out.usage.input_tokens if out.usage else None, "out": out.usage.output_tokens if out.usage else None,
            "stop": out.stop_reason,
        }

    results = await asyncio.gather(*(one(i, *c) for i, c in enumerate(cases)))
    for r in results:
        print(json.dumps(r, ensure_ascii=False))

    sanity = await model.generate(
        input=[ChatMessageUser(content="Compute 41.7 minus 38.2, and 126.5 times 2. Reply with the two numbers only, separated by a comma.")],
        config=cfg.generate_config(max_tokens=30),
    )
    ans = sanity.completion.replace(" ", "")
    sanity_ok = "3.5" in ans and "253" in ans
    n_ok = sum(r["ok"] for r in results)
    print(f"\nTOOL CALLS OK: {n_ok}/{len(results)}")
    print(f"SANITY: {sanity.completion!r} -> {'ok' if sanity_ok else 'FAIL'}")
    passed = n_ok == len(results) and sanity_ok
    print("GATE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
