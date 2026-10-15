"""Answer a question from company documents, with citations that are checked.

    retrieve (LlamaIndex, hybrid) -> the model answers ONLY from the passages, citing [D1]... or says
    NOT_IN_DOCUMENTS -> code verifies every figure against the passages and every citation against the
    passage it names.

A citation is not trusted just because it is present: the cited passage must exist, and the cited sentence
must share its content words with that passage (a heuristic, reported as such).
"""
import re

from agent.verify import extract_numbers, verify_answer

NOT_FOUND = "NOT_IN_DOCUMENTS"
SYSTEM = ("You answer questions about a company using ONLY the numbered passages you are given. "
          "You never use outside knowledge.")
_CITE = re.compile(r"\[D(\d+)\]")
_STOP = set("""the a an and or of to in on for with from by is are was were be been it its this that these those as at
which who what when where how than then their there they we our us you your has have had not no do does did will
would can could should may might also into over under about after before during while per more most less very""".split())


def passages_prompt(question, passages):
    blocks = "\n\n".join(f"[D{i}] ({p['source']}, page {p['page']})\n{p['text']}" for i, p in enumerate(passages, 1))
    return f"""PASSAGES:
{blocks}

QUESTION: {question}

Rules:
- Answer in 1 to 3 sentences using only the passages.
- Put the citation [D1], [D2]... right after every claim it supports.
- Copy figures exactly as written in the passages.
- If the passages do not contain the answer, reply with exactly: {NOT_FOUND}
- Do not guess, estimate or use anything you remember about the company."""


def _content_tokens(text):
    return {t for t in re.findall(r"[a-z0-9$%.]+", text.lower()) if len(t) > 2 and t not in _STOP}


def citation_report(answer, passages):
    """Per cited sentence: do the cited passages exist, and do they plausibly support it?"""
    out = []
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    for sent in parts:
        ids = [int(i) for i in _CITE.findall(sent)]
        if not ids:
            continue
        claim = _content_tokens(_CITE.sub("", sent))
        missing = [i for i in ids if not 1 <= i <= len(passages)]
        support = 0.0
        for i in ids:
            if 1 <= i <= len(passages):
                have = _content_tokens(passages[i - 1]["text"])
                if claim:
                    support = max(support, len(claim & have) / len(claim))
        out.append({"sentence": sent, "cited": ids, "missing": missing, "support": round(support, 2),
                    "ok": not missing and support >= 0.5})
    return out


def answer_question(question, index, llm, symbol=None, k=3):
    """Returns {answer, abstained, passages, citations, verification, model}. `llm` is a FallbackLLM-like object."""
    passages = index.search(question, k=k, symbol=symbol)
    if not passages:
        return {"answer": NOT_FOUND, "abstained": True, "passages": [], "citations": [], "verification": None,
                "model": None, "note": "no passages were retrieved"}
    text, meta = llm.generate(passages_prompt(question, passages), system=SYSTEM)
    text = text.strip()
    abstained = text.upper().startswith(NOT_FOUND)
    ledger = [{"tool": "search_documents", "args": {}, "result": {
        "passages": [{"id": f"D{i}", "text": p["text"]} for i, p in enumerate(passages, 1)]}}]
    figures = verify_answer(text, ledger) if not abstained else None
    cites = [] if abstained else citation_report(text, passages)
    uncited = (not abstained) and not cites and bool(extract_numbers(text) or len(text.split()) > 3)
    return {"answer": text, "abstained": abstained, "passages": passages, "citations": cites,
            "verification": figures, "uncited": uncited, "model": f"{meta['provider']}/{meta['model']}",
            "grounded": (not abstained) and not uncited and all(c["ok"] for c in cites)
                        and (figures is None or figures["n_problems"] == 0)}
