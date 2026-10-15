"""Evaluate the document search and the grounded answers against ground truth read from the transcript.

Ground truth: the TCS Q4 FY26 earnings-call transcript (NSE filing). Each answerable question lists
regexes that must all appear in the passage that answers it; the unanswerable ones are facts that are
NOT in the document, to test that the system abstains instead of guessing.

    python -m pipeline.rag_eval              # retrieval only (free, no model)
    python -m pipeline.rag_eval --answers    # + 2 model requests: with documents vs without
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (question, [regexes that must ALL appear in the answering passage], PDF page where it is found)
ANSWERABLE = [
    ("What total contract value did TCS book in FY26 and how many mega deals?", [r"40\.7", r"5 mega deals"], 5),
    ("How much did TCS revenue change in FY26 in constant currency?", [r"declined by 2\.4%"], 5),
    ("What was TCS's operating margin for FY26?", [r"operating margin was 25%", r"70 basis points"], 8),
    ("How strong was cash conversion in Q4?", [r"106\.7%"], 7),
    ("What final dividend did the board recommend?", [r"₹31 per share", r"₹110"], 8),
    ("What was TCS's global headcount at the end of March 2026?", [r"584,519"], 13),
    ("What is TCS's annualised AI revenue?", [r"\$2\.3 billion"], 8),
    ("Does TCS give specific revenue or earnings guidance?", [r"don't provide any speciﬁc revenue or earnings guidance"], 3),
    ("How many days of sales outstanding (DSO) did TCS report for Q4?", [r"74 DSO"], 7),
    ("How many mega deals were booked in the fourth quarter?", [r"three mega deals"], 21),
]
UNANSWERABLE = [
    "What was TCS's employee attrition rate in Q4?",
    "Is TCS planning a share buyback and for how much?",
    "What is the analyst target price for TCS shares?",
    "What operating margin did Infosys report for FY26?",
]
NOT_FOUND = "NOT_IN_DOCUMENTS"


def retrieval_report(index, symbol="TCS", k=3):
    """Hit@k for meaning-only versus hybrid retrieval. No model needed."""
    rows = []
    for q, pats, page in ANSWERABLE:
        row = {"question": q, "page": page}
        for mode, hybrid in (("meaning", False), ("hybrid", True)):
            hits = index.search(q, k=k, symbol=symbol, hybrid=hybrid)
            found = [h for h in hits if all(re.search(p, h["text"], re.I) for p in pats)]
            row[mode] = bool(found)
            row[f"{mode}_rank"] = next((i + 1 for i, h in enumerate(hits) if h in found), None)
        rows.append(row)
    n = len(rows)
    return {"k": k, "n": n, "meaning_hits": sum(r["meaning"] for r in rows),
            "hybrid_hits": sum(r["hybrid"] for r in rows), "rows": rows}


def absent_in_document(index, patterns=(r"attrition", r"buy-?back", r"target price", r"infosys")):
    """Confirms the 'unanswerable' facts really are absent by scanning every passage."""
    text = " ".join(n.get_content() for n in index._nodes).lower()
    return {p: bool(re.search(p, text)) for p in patterns}


# ---------------------------------------------------------------- batched generation (2 requests in total)

def batch_prompt(questions_with_passages=None, questions=None):
    """One request that answers every question; with passages when given, from memory when not."""
    if questions_with_passages is not None:
        blocks = []
        for i, (q, passages) in enumerate(questions_with_passages, 1):
            ps = "\n".join(f"  [D{j}] (page {p['page']}) {p['text']}" for j, p in enumerate(passages, 1))
            blocks.append(f"QUESTION {i}: {q}\nPASSAGES FOR QUESTION {i}:\n{ps}")
        rules = (f"Answer each question using ONLY its own passages. Cite like [D1]. If a question's passages do not "
                 f"contain the answer, the answer must be exactly {NOT_FOUND}. Never use outside knowledge.")
    else:
        blocks = [f"QUESTION {i}: {q}" for i, q in enumerate(questions, 1)]
        rules = (f"Answer each question about Tata Consultancy Services' Q4 FY26 earnings call from your own knowledge. "
                 f"If you do not know, answer exactly {NOT_FOUND}.")
    return ("\n\n".join(blocks) + f"\n\n{rules}\nReply with JSON only: "
            '{"answers": [{"n": 1, "answer": "..."}, ...]} with one entry per question, in order.')


def parse_answers(text, n):
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    try:
        items = json.loads(body[body.index("{"):body.rindex("}") + 1])["answers"]
        by_n = {int(i.get("n", 0)): str(i.get("answer", "")) for i in items if isinstance(i, dict)}
    except (ValueError, KeyError, TypeError):
        by_n = {}
    return [by_n.get(i, "") for i in range(1, n + 1)]


def score_answer(answer, patterns):
    """Answerable: every pattern must appear in the answer (numbers copied correctly)."""
    return bool(answer) and NOT_FOUND not in answer.upper() and all(re.search(p, answer, re.I) for p in patterns)


def score_abstention(answer):
    return NOT_FOUND in (answer or "").upper() or re.search(r"\b(not (?:mentioned|stated|provided|disclosed|available)|"
                                                            r"no (?:information|mention)|don't know|do not know)\b",
                                                            answer or "", re.I) is not None


def generation_report(index, llm, symbol="TCS", k=3):
    all_q = [q for q, _, _ in ANSWERABLE] + UNANSWERABLE
    with_docs = [(q, index.search(q, k=k, symbol=symbol)) for q in all_q]
    results = {}
    for name, prompt in (("with_documents", batch_prompt(questions_with_passages=with_docs)),
                         ("without_documents", batch_prompt(questions=all_q))):
        text, meta = llm.generate(prompt, system="You answer precisely and never invent figures.")
        answers = parse_answers(text, len(all_q))
        rows = []
        for i, (q, pats, _) in enumerate(ANSWERABLE):
            rows.append({"question": q, "answer": answers[i], "correct": score_answer(answers[i], pats)})
        for j, q in enumerate(UNANSWERABLE):
            a = answers[len(ANSWERABLE) + j]
            rows.append({"question": q, "answer": a, "abstained": score_abstention(a), "invented": bool(a) and not score_abstention(a)})
        results[name] = {"model": f"{meta['provider']}/{meta['model']}", "rows": rows,
                         "correct": sum(r.get("correct", False) for r in rows[:len(ANSWERABLE)]),
                         "answerable": len(ANSWERABLE),
                         "abstained": sum(r.get("abstained", False) for r in rows[len(ANSWERABLE):]),
                         "invented": sum(r.get("invented", False) for r in rows[len(ANSWERABLE):]),
                         "unanswerable": len(UNANSWERABLE)}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also make 2 model requests (with / without documents)")
    ap.add_argument("--knowledge", default=str(ROOT / "knowledge"))
    a = ap.parse_args()
    from pipeline.rag import DocumentIndex
    idx = DocumentIndex.build(a.knowledge)
    print(f"index: {idx.n_docs} pages, {idx.n_chunks} passages, backend {idx.backend}")
    print("absent-fact check (True means the term IS in the document):", absent_in_document(idx))
    r = retrieval_report(idx)
    print(f"\nRETRIEVAL  hit@{r['k']} over {r['n']} answerable questions")
    print(f"  meaning only : {r['meaning_hits']}/{r['n']}")
    print(f"  hybrid       : {r['hybrid_hits']}/{r['n']}")
    for row in r["rows"]:
        print(f"   {'✓' if row['hybrid'] else '✗'} (meaning {'✓' if row['meaning'] else '✗'}) p.{row['page']:<3} {row['question']}")
    if a.answers:
        from pipeline.llm import FallbackLLM
        g = generation_report(idx, FallbackLLM())
        print("\nANSWERS")
        for name, v in g.items():
            print(f"  {name:18} model {v['model']}: correct {v['correct']}/{v['answerable']}, "
                  f"abstained on {v['abstained']}/{v['unanswerable']} unanswerable, invented {v['invented']}")
        out = ROOT / "reports" / "rag_eval.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"retrieval": r, "generation": g}, indent=1, ensure_ascii=False))
        print("\ndetails:", out)


if __name__ == "__main__":
    main()
