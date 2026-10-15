"""The research agent: a bounded loop in which the MODEL decides what to look into.

Two decisions are the model's, and everything else is code:

  1. triage      Given each stock's rule flags, which stocks are worth investigating, and what question
                 should be answered for each? (falls back to the highest rule scores if the model fails)
  2. investigate For one stock, which tool to call next: search the documents, read the news, look at the
                 fundamentals history, or check the risk profile. It may stop early.

Bounds, because this runs on a free quota: at most `max_stocks` stocks and `max_tool_calls` tool calls per
stock (plus one closing call), so a full run is at most 1 + max_stocks * (max_tool_calls + 1) requests.

Every tool result becomes numbered evidence [E1], [E2]... The agent's findings must cite it, and each
finding is then checked: the cited evidence must exist, the finding must share its content words with that
evidence, and every figure must trace to a tool result.
"""
import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from agent import ledger
from agent.verify import verify_answer
from pipeline.llm import AllModelsFailed
from pipeline.qa import _content_tokens
from pipeline.rules import concern_score, top_concerns

TOOL_HELP = {
    "search_documents": ('{"query": "<what to look for>"}',
                         "Search this company's filings and earnings-call transcripts by meaning. Returns cited passages."),
    "get_news": ("{}", "Recent news headlines about this company (last 7 days)."),
    "get_fundamentals_history": ("{}", "Revenue, margins, ROCE, ROE and free cash flow for the last few fiscal years."),
    "get_risk_profile": ("{}", "Volatility, beta, 1-year drawdown, gap to the 200-day average and earnings timing."),
}
# A statement of ABSENCE ("the evidence does not explain...") is not a positive claim and must not be graded
# as one: it has no content for a citation to support.
_GAP = re.compile(r"^\s*(no |none |nothing |there is no |there are no )|"
                  r"\b(?:do(?:es)? not|did not|don't|doesn't|cannot|can't|not enough|insufficient|no (?:evidence|information|"
                  r"data|mention|explanation))\b.{0,60}\b(?:contain|explain|provide|show|mention|include|indicate|specify|"
                  r"state|address|support|evidence|information|data|available|sufficient)\b|\bno evidence\b", re.I)


@dataclass
class Tools:
    """What the agent can reach. Each callable takes the stock symbol (and a query for search)."""
    docs: Optional[Callable] = None            # (query, symbol) -> [{source, page, text, ...}]
    news: Optional[Callable] = None            # (symbol) -> {"headlines": [{"date","title"}]}
    fundamentals: Optional[Callable] = None    # (symbol) -> get_fundamentals-style dict
    risk: Optional[Callable] = None            # (symbol) -> get_risk-style dict


# --------------------------------------------------------------------------- parsing

def _json_object(text):
    if not text:
        return None
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(body[body.index("{"):body.rindex("}") + 1])
    except (ValueError, TypeError):
        return None


def parse_action(text):
    """-> {"action", "args", "why", "findings"} or None when the reply is unusable."""
    obj = _json_object(text)
    if not isinstance(obj, dict):
        return None
    action = str(obj.get("action", "")).strip()
    if action != "finish" and action not in TOOL_HELP:
        return None
    findings = [f for f in (obj.get("findings") or []) if isinstance(f, dict)]
    return {"action": action, "args": obj.get("args") if isinstance(obj.get("args"), dict) else {},
            "why": str(obj.get("why", "")).strip(), "findings": findings}


def parse_triage(text, allowed, max_stocks):
    obj = _json_object(text) or {}
    picks, seen = [], set()
    for item in obj.get("investigate", []) if isinstance(obj, dict) else []:
        sym = str(item.get("symbol", "")).upper().strip() if isinstance(item, dict) else ""
        if sym in allowed and sym not in seen:
            picks.append({"symbol": sym, "question": str(item.get("question", "")).strip()})
            seen.add(sym)
    return picks[:max_stocks]


# --------------------------------------------------------------------------- triage

def default_question(symbol, flags):
    concerns = [f for f in flags if f.kind == "concern"]
    if not concerns:
        return f"Is there anything in the documents or news that should change the view on {symbol}?"
    parts = ", ".join(f.title.lower() for f in sorted(concerns, key=lambda f: -len(f.detail))[:3])
    return f"What explains these concerns for {symbol}: {parts}? Is it temporary or structural?"


def triage(llm, flags_by_symbol, companies, max_stocks=2, has_docs=None):
    """The model picks which flagged stocks to investigate and what to ask. Returns (picks, requests, how).

    `has_docs` is the set of symbols that have filings or transcripts. The model is told which do, because a
    live run showed it investigating stocks it could not research (and ignoring the one it could). If it still
    picks none of them while a flagged stock has documents, one is swapped in and the trace says so.
    """
    has_docs = set(has_docs or [])
    candidates = [s for s, f in flags_by_symbol.items() if concern_score(f) > 0]
    if not candidates or max_stocks <= 0:
        return [], 0, "nothing flagged"
    lines = []
    for s in candidates:
        fl = flags_by_symbol[s]
        docs = "filings and transcripts available" if s in has_docs else "no filings available"
        lines.append(f"{s} ({companies.get(s, s)}), concern score {concern_score(fl)}, {docs}:")
        lines += [f"   - [{f.severity}] {f.title}: {f.detail}" for f in fl if f.kind == "concern"]
    prompt = (f"Rule-based flags for a portfolio review:\n" + "\n".join(lines) +
              f"\n\nChoose at most {max_stocks} stocks that most deserve a closer look by a research analyst, and for each "
              f"write the single most useful question to answer. Prefer stocks where the explanation is not obvious from "
              f"the flags themselves, and where research is possible: stocks marked 'no filings available' can only be "
              f"researched through news and statements. Reply with JSON only: "
              f'{{"investigate": [{{"symbol": "...", "question": "..."}}]}}')
    try:
        text, _ = llm.generate(prompt, system="You are a careful senior analyst deciding where to spend research time.")
    except AllModelsFailed:
        text = ""
    picks = parse_triage(text, set(candidates), max_stocks)
    how = "model"
    if not picks:                                    # unusable or unavailable: fall back to the rule scores
        how = "rule scores (model gave no usable answer)"
        picks = [{"symbol": s, "question": ""} for s in top_concerns(flags_by_symbol, max_stocks)]
    ranked = top_concerns(flags_by_symbol, len(flags_by_symbol))
    with_docs = [s for s in ranked if s in has_docs]
    if with_docs and picks and not any(p["symbol"] in has_docs for p in picks):
        added = with_docs[0]
        picks = picks[:max(1, max_stocks) - 1] + [{"symbol": added, "question": ""}]
        how += f"; {added} added because it is the flagged stock that has documents"
    for p in picks:
        if not p["question"]:
            p["question"] = default_question(p["symbol"], flags_by_symbol[p["symbol"]])
    return picks, 1, how


# --------------------------------------------------------------------------- one investigation

def _fundamentals_text(result):
    data = (result or {}).get("data", {})
    rows = []
    for h in data.get("history_newest_first") or data.get("history") or []:
        bits = [f"revenue ₹{h['revenue_cr']:,.0f} crore" if h.get("revenue_cr") is not None else None,
                f"EBIT margin {h['ebit_margin_pct']}%" if h.get("ebit_margin_pct") is not None else None,
                f"ROCE {h['roce_pct']}%" if h.get("roce_pct") is not None else None,
                f"ROE {h['roe_pct']}%" if h.get("roe_pct") is not None else None,
                f"debt-to-equity {h['de_ratio']}" if h.get("de_ratio") is not None else None,
                f"free cash flow ₹{h['fcf_cr']:,.0f} crore" if h.get("fcf_cr") is not None else None]
        rows.append(f"FY ending {h.get('fy')}: " + ", ".join(b for b in bits if b))
    na = data.get("not_applicable")
    if na:
        rows.append("Not applicable for this kind of company: " + ", ".join(na))
    return "\n".join(rows) or "No fundamentals history available."


def _risk_text(result):
    d = (result or {}).get("data", {})
    parts = [("volatility", d.get("volatility_pct"), "%"), ("beta against NIFTY 50", d.get("beta"), ""),
             ("maximum drawdown over 1 year", d.get("max_drawdown_1y_pct"), "%"),
             ("price versus its 200-day average", d.get("pct_vs_200dma"), "%")]
    txt = ", ".join(f"{n} {v}{u}" for n, v, u in parts if v is not None)
    if d.get("earnings_proximity"):
        txt += f"; earnings: {d['earnings_proximity']}"
    return txt or "No risk profile available."


def investigate_stock(llm, tools, symbol, company, sector, question, flags, max_tool_calls=2, ledger_view=None,
                      doc_sources=None):
    """Run the bounded loop for one stock. Returns a dict describing everything that happened."""
    ledger_view = ledger_view or ledger.snapshot
    evidence, steps, requests = [], [], 0
    used_calls, stopped, findings_raw = 0, "budget", []

    def add_evidence(tool, text, source=None, page=None):
        eid = f"E{len(evidence) + 1}"
        evidence.append({"id": eid, "tool": tool, "text": text, "source": source, "page": page})
        return eid

    def render():
        if not evidence:
            return "(none yet)"
        out = []
        for e in evidence:
            tag = e["tool"] + (f", {e['source']} p.{e['page']}" if e.get("source") else "")
            out.append(f"[{e['id']}] ({tag}) {e['text'][:600]}")
        return "\n".join(out)

    flag_lines = "\n".join(f"- {f.title}: {f.detail}" for f in flags if f.kind == "concern") or "- (no concerns raised)"
    strengths = "\n".join(f"- {f.title}: {f.detail}" for f in flags if f.kind == "strength")
    offered = {n: v for n, v in TOOL_HELP.items() if n != "search_documents" or doc_sources}
    tool_lines = "\n".join(f'- {n} {a}: {d}' for n, (a, d) in offered.items())
    docs_line = ("DOCUMENTS AVAILABLE FOR THIS COMPANY: " + ", ".join(doc_sources)) if doc_sources else \
                "No filings or transcripts are available for this company, so do not search documents."

    while True:
        left = max_tool_calls - used_calls
        prompt = f"""You are a research analyst investigating ONE stock for a human reviewer.
STOCK: {symbol} ({company}), sector {sector}
QUESTION: {question}
WHY IT WAS SELECTED (rule flags):
{flag_lines}
{('STRENGTHS NOTED:' + chr(10) + strengths) if strengths else ''}
{docs_line}
EVIDENCE SO FAR:
{render()}

TOOLS:
{tool_lines}
You have {left} tool call(s) left.
Reply with ONE JSON object, nothing else.
To use a tool: {{"action": "<tool name>", "args": <args>, "why": "<one sentence>"}}
To finish{' (you MUST finish now)' if left <= 0 else ''}: {{"action": "finish", "findings": [{{"claim": "...", "evidence": ["E1"]}}]}}
Findings rules: 1 to 3 findings. A claim must be supported by the evidence you cite, with figures copied exactly.
If the evidence does not explain the concern, say so as a finding with an empty evidence list."""
        try:
            text, meta = llm.generate(prompt, system="You investigate methodically and never state anything the evidence does not support.")
        except AllModelsFailed:
            stopped = "model_unavailable"
            break
        requests += 1
        act = parse_action(text)
        if act is None:
            steps.append({"n": len(steps) + 1, "action": "unusable_reply", "args": {}, "why": "", "evidence": []})
            if requests >= max_tool_calls + 2:
                stopped = "unusable_replies"
                break
            continue
        if act["action"] == "finish" or left <= 0:
            findings_raw = act["findings"]
            stopped = "finished"
            steps.append({"n": len(steps) + 1, "action": "finish", "args": {}, "why": act["why"], "evidence": []})
            break

        used_calls += 1
        name, args, new_ids = act["action"], act["args"], []
        try:
            if name == "search_documents" and not doc_sources:
                new_ids = [add_evidence(name, "No documents are available for this company.")]
            elif name == "search_documents" and tools.docs:
                q = str(args.get("query") or question)
                hits = tools.docs(q, symbol) or []
                ledger.record("search_documents", {"query": q, "symbol": symbol}, {
                    "symbol": symbol, "passages": [{"text": h["text"], "source": h.get("source"), "page": h.get("page")} for h in hits]})
                new_ids = [add_evidence(name, h["text"], h.get("source"), h.get("page")) for h in hits]
                if not hits:
                    new_ids = [add_evidence(name, f"No passages found for '{q}'.")]
            elif name == "get_news" and tools.news:
                r = tools.news(symbol)
                heads = (r or {}).get("headlines") or []
                new_ids = [add_evidence(name, f"{h.get('title')} ({h.get('date')})") for h in heads[:5]]
                if not heads:
                    new_ids = [add_evidence(name, "No headlines in the last 7 days.")]
            elif name == "get_fundamentals_history" and tools.fundamentals:
                new_ids = [add_evidence(name, _fundamentals_text(tools.fundamentals(symbol)))]
            elif name == "get_risk_profile" and tools.risk:
                new_ids = [add_evidence(name, _risk_text(tools.risk(symbol)))]
            else:
                new_ids = [add_evidence(name, "This tool is not available in this run.")]
        except Exception as e:                                           # a failing tool must not end the run
            new_ids = [add_evidence(name, f"The tool failed: {type(e).__name__}")]
        steps.append({"n": len(steps) + 1, "action": name, "args": args, "why": act["why"], "evidence": new_ids})

    findings = check_findings(findings_raw, evidence, ledger_view())
    return {"symbol": symbol, "company": company, "question": question, "steps": steps, "evidence": evidence,
            "findings": findings, "requests": requests, "tool_calls": used_calls, "stopped": stopped}


# --------------------------------------------------------------------------- checking findings

def check_findings(raw, evidence, ledger_entries):
    """Each finding: do its citations exist, do they support the claim, and do its figures trace to a tool."""
    by_id = {e["id"]: e for e in evidence}
    out = []
    for f in raw[:3]:
        claim = str(f.get("claim", "")).strip()
        cited = [str(e).strip().strip("[]") for e in (f.get("evidence") or []) if str(e).strip()]
        issues = []
        if not claim:
            continue
        missing = [c for c in cited if c not in by_id]
        if missing:
            issues.append(f"cites evidence that does not exist: {', '.join(missing)}")
        is_gap = bool(_GAP.search(claim))
        if not cited:
            if not is_gap:
                issues.append("makes a claim without citing any evidence")
            kind = "gap"
        elif is_gap:
            kind = "gap"                      # an absence statement, even if it points at the evidence it checked
        else:
            kind = "claim"
            have = set()
            for c in cited:
                if c in by_id:
                    have |= _content_tokens(by_id[c]["text"])
            want = _content_tokens(claim)
            support = len(want & have) / len(want) if want else 0.0
            if want and support < 0.4:
                issues.append(f"the cited evidence does not appear to support it (word overlap {support:.0%})")
        rep = verify_answer(claim, ledger_entries)
        for p in rep["problems"]:
            if kind == "gap" and p.get("kind") != "number":
                continue            # "the documents do not explain the decline" is not a claim that ROCE is declining
            issues.append(f"{'figure' if p.get('kind') == 'number' else p.get('kind', 'claim')} {p['claim']} is {p['status']}")
        out.append({"claim": claim, "evidence": cited, "kind": kind, "verified": not issues, "issues": issues})
    return out
