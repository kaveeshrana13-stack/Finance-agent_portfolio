"""A language model with a free fallback chain: Gemini -> a second Gemini model -> local Ollama.

Each report needs only a couple of model calls, because code (not the model) gathers the data.
The free Gemini tier is small (20 requests per day per model), so when one model's daily quota is
exhausted the next is tried, and a model that returned a daily-quota error is skipped for the rest of
the process instead of being retried on every call.
"""
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHAIN = [
    ("gemini", "gemini-2.5-flash"),
    ("gemini", "gemini-flash-lite-latest"),
    ("ollama", "qwen2.5-coder:7b"),
]


class AllModelsFailed(RuntimeError):
    def __init__(self, attempts):
        self.attempts = attempts
        super().__init__("no model in the chain could answer: " + "; ".join(
            f"{a['provider']}/{a['model']}: {a['error']}" for a in attempts))


def _load_gemini_key():
    """Expose the project's GOOGLE_API_KEY to LiteLLM as GEMINI_API_KEY. Never prints it."""
    if os.getenv("GEMINI_API_KEY"):
        return True
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            from dotenv import dotenv_values
            key = dotenv_values(PROJECT_ROOT / ".env").get("GOOGLE_API_KEY")
        except ImportError:
            key = None
    if key:
        os.environ["GEMINI_API_KEY"] = key
    return bool(key)


def _litellm_complete(provider, model, messages, timeout, temperature):
    import litellm
    litellm.suppress_debug_info = True
    if provider == "gemini":
        if not _load_gemini_key():
            raise RuntimeError("GOOGLE_API_KEY not found")
        kwargs = {"model": f"gemini/{model}", "max_tokens": 4096}
    else:
        kwargs = {"model": f"ollama_chat/{model}", "api_base": "http://127.0.0.1:11434", "max_tokens": 1500}
    r = litellm.completion(messages=messages, temperature=temperature, timeout=timeout, num_retries=0, **kwargs)
    text = r.choices[0].message.content
    if not text:
        raise RuntimeError("empty reply")
    return text


class FallbackLLM:
    """generate(prompt) -> (text, meta). Tries each model in order; records every attempt."""

    def __init__(self, chain=None, complete=None, timeout=90, temperature=0.2):
        self.chain = list(chain or DEFAULT_CHAIN)
        self._complete = complete or _litellm_complete
        self.timeout, self.temperature = timeout, temperature
        self.attempts = []                  # every try across the whole run, for the report
        self._exhausted = set()             # (provider, model) that hit a daily quota

    def generate(self, prompt, system=None):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        failures = []
        for provider, model in self.chain:
            if (provider, model) in self._exhausted:
                continue
            started = time.time()
            try:
                text = self._complete(provider, model, messages, self.timeout, self.temperature)
                self.attempts.append({"provider": provider, "model": model, "ok": True,
                                      "seconds": round(time.time() - started, 1), "error": None})
                return text, {"provider": provider, "model": model, "fallbacks_used": len(failures)}
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)}"
                key = os.environ.get("GEMINI_API_KEY", "")
                msg = (msg.replace(key, "<key>") if key else msg).replace("\n", " ")
                if "PerDay" in msg or "RESOURCE_EXHAUSTED" in msg:
                    self._exhausted.add((provider, model))
                rec = {"provider": provider, "model": model, "ok": False,
                       "seconds": round(time.time() - started, 1), "error": msg[:200]}
                self.attempts.append(rec)
                failures.append(rec)
        raise AllModelsFailed(failures or self.attempts[-len(self.chain):])
