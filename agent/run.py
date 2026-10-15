"""Run the finance agent on a local Ollama model.

    python -m agent.run "What is the current market regime?"
    python -m agent.run --model qwen2.5-coder:7b --steps 8 "Is Titan's ROCE improving?"
"""
import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smolagents import CodeAgent, LiteLLMModel

from agent import ledger
from agent.tools import ALL_TOOLS, AGENT_TOOLS
from agent.verify import annotate, summarize_report, verify_answer

TOOL_NAMES = [t.name for t in ALL_TOOLS]

GUIDELINES = """You are a research assistant for Indian equities. Rules:
- Every number you state must come from a tool result. Never use remembered figures.
- If a tool returns warnings, status 'partial', 'not_applicable' or an empty result, say so plainly.
- Name the tool(s) you used. Units: *_pct are percent, *_cr are INR crore, de_ratio is a plain ratio.
- Pass the company name EXACTLY as the user wrote it. Never substitute a ticker you remember; tickers change.
- If a result has ok=False with a message (e.g. ambiguous company), give that message and stop.
- Tool results are dicts. Check result["ok"] (True/False) to see whether the data is usable, and read
  fields such as result["data"]["roce_pct"] directly. Do not invent status strings.
- Before final_answer, check that every part of the question is answered (e.g. "compare A and B on X and Y"
  needs X and Y for both A and B). Finish by calling final_answer(...) with a plain-text string.
- If a tool returns ok=False with a message, give the user that message instead of guessing.
- Never write "not available" for a value unless a tool you actually called returned it as None or ok=False.
  If you have not called the tool that provides it, call it.
- As soon as a tool result contains what you need, call final_answer with it immediately. Do not call
  more tools "to double check" and do not repeat a tool call that already succeeded.
- Give a short, direct answer. This is research support, not investment advice."""


def build_agent(model_id="qwen2.5-coder:7b", base_url="http://127.0.0.1:11434", max_steps=8, num_ctx=8192,
                verbosity=1, temperature=0.1):
    """Low temperature: a tool-using agent should be consistent, not creative."""
    model = LiteLLMModel(model_id=f"ollama_chat/{model_id}", api_base=base_url, num_ctx=num_ctx,
                         temperature=temperature)
    return CodeAgent(tools=AGENT_TOOLS, model=model, max_steps=max_steps, verbosity_level=verbosity,
                     additional_authorized_imports=["json", "math"])


def clean_answer(text):
    """Drop 'Thought: ...' scaffolding the model sometimes leaks before 'Final answer:'."""
    if not text:
        return text
    m = list(re.finditer(r"final answer\s*:\s*", text, re.I))
    if m and re.match(r"\s*thought\s*:", text, re.I):
        text = text[m[-1].end():]
    return re.sub(r"</?code>\s*$", "", text).strip()


MAX_SECONDS = 300      # one question; a single attempt once ran 936 s


class _TimeLimit(Exception):
    pass


def ask(agent, question, max_seconds=MAX_SECONDS):
    """Run one question; return the answer plus what the agent actually did.

    A wall-clock limit (Unix, main thread) stops a stalled model; the result then has error set.
    """
    ledger.reset()
    started = time.time()
    error = None

    def _alarm(signum, frame):
        raise _TimeLimit(f"no answer within {max_seconds}s")

    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(max_seconds)
    try:
        answer = agent.run(f"{GUIDELINES}\n\nQuestion: {question}", reset=True)
    except Exception as e:                      # keep the harness alive; report the failure
        answer, error = None, f"{type(e).__name__}: {e}"
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    steps = list(agent.memory.steps)
    code = [getattr(s, "code_action", None) or "" for s in steps if hasattr(s, "tool_calls") or hasattr(s, "code_action")]
    used = sorted({n for c in code for n in TOOL_NAMES if re.search(rf"\b{n}\s*\(", c)})
    return {"question": question, "answer": None if answer is None else clean_answer(str(answer)), "error": error,
            "seconds": round(time.time() - started, 1), "steps": len([c for c in code if c]),
            "tools_called": used, "code": code, "ledger": ledger.snapshot()}


def ask_verified(agent, question, retry=True):
    """ask() plus verification. One corrective retry when the answer contains unsupported figures.

    Returns the final result dict with 'report' (the verifier's verdict), 'answer_annotated'
    (problem figures tagged), and 'attempts' (every try, so nothing is hidden).
    """
    first = ask(agent, question)
    first["report"] = verify_answer(first["answer"], first["ledger"])
    attempts = [first]
    final = first

    if retry and first["answer"] and first["report"]["problems"]:
        bad = ", ".join(f"{c['claim']} ({c['status']})" for c in first["report"]["problems"][:6])
        nudge = (f"{question}\n\nYour previous answer contained figures that no tool returned or that sit under "
                 f"the wrong label: {bad}. Answer again. Use only numbers from tool results; if a value was not "
                 f"returned by a tool, say it is not available instead of stating a number.")
        second = ask(agent, nudge)
        second["report"] = verify_answer(second["answer"], second["ledger"])
        attempts.append(second)
        if second["answer"] and second["report"]["n_problems"] <= first["report"]["n_problems"]:
            final = second

    out = dict(final)
    out["answer_annotated"] = annotate(final["answer"] or "", final["report"]) if final["answer"] else None
    out["attempts"] = [{"answer": a["answer"], "verdict": a["report"]["verdict"],
                        "problems": [c["claim"] for c in a["report"]["problems"]], "seconds": a["seconds"]}
                       for a in attempts]
    out["verification"] = summarize_report(final["report"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args()
    result = ask_verified(build_agent(a.model, max_steps=a.steps, verbosity=a.verbosity), a.question)
    if a.json:
        print(json.dumps({k: v for k, v in result.items() if k not in ("code", "ledger")}, indent=1,
                         ensure_ascii=False, default=str))
    else:
        print(f"\nANSWER: {result['answer_annotated'] or result['answer']}")
        print(f"VERIFICATION: {result['verification']}")
        for c in result["report"]["problems"]:
            print(f"  ! {c['claim']}: {c['status']} ({c['source']})")
        print(f"[{result['seconds']}s, {result['steps']} steps, tools: {result['tools_called']}, "
              f"attempts: {len(result['attempts'])}]")


if __name__ == "__main__":
    main()
