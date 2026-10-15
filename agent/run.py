"""Run the finance agent on a local Ollama model.

    python -m agent.run "What is the current market regime?"
    python -m agent.run --model qwen2.5-coder:7b --steps 8 "Is Titan's ROCE improving?"
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smolagents import CodeAgent, LiteLLMModel

from agent.tools import ALL_TOOLS, AGENT_TOOLS

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


def ask(agent, question):
    """Run one question; return the answer plus what the agent actually did."""
    started = time.time()
    error = None
    try:
        answer = agent.run(f"{GUIDELINES}\n\nQuestion: {question}", reset=True)
    except Exception as e:                      # keep the harness alive; report the failure
        answer, error = None, f"{type(e).__name__}: {e}"
    steps = list(agent.memory.steps)
    code = [getattr(s, "code_action", None) or "" for s in steps if hasattr(s, "tool_calls") or hasattr(s, "code_action")]
    used = sorted({n for c in code for n in TOOL_NAMES if re.search(rf"\b{n}\s*\(", c)})
    return {"question": question, "answer": None if answer is None else str(answer), "error": error,
            "seconds": round(time.time() - started, 1), "steps": len([c for c in code if c]),
            "tools_called": used, "code": code}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args()
    result = ask(build_agent(a.model, max_steps=a.steps, verbosity=a.verbosity), a.question)
    print(json.dumps(result, indent=1, ensure_ascii=False) if a.json else
          f"\nANSWER: {result['answer']}\n[{result['seconds']}s, {result['steps']} steps, tools: {result['tools_called']}]")


if __name__ == "__main__":
    main()
