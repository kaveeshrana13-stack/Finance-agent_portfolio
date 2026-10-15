"""The weekly report as a LangGraph graph.

    resolve -> market -> stock_data -> portfolio -> rules -> memory_load -> investigate
         -> write_notes <-> check_notes -> guardrails -> compare -> write_summary <-> check_summary
         -> save_memory -> assemble

Code gathers every fact (no model involved) and records it in the ledger. Rules turn facts into FLAGS;
flags decide what the research agent (the `investigate` node) looks into. The model writes at three
places (investigate, write_notes, write_summary) and everything it writes is verified. A note with a
figure no tool returned is re-run once with the problems listed; whatever is still unsupported is tagged
[unverified]. Hard rules act as guardrails: the model may be more cautious than a rule, never less.
"""
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agent import ledger
from agent.verify import annotate, verify_answer
from pipeline import investigator as inv
from pipeline import memory as mem
from pipeline import rules as rl
from pipeline.llm import AllModelsFailed
from tools.portfolio import summarize_portfolio
from tools.stocks import volatility_bucket

ACTIONS = ("HOLD", "ADD", "REDUCE", "EXIT", "REVIEW")
MAX_ATTEMPTS = 2          # first try + one corrective retry, per model-written section

SYSTEM = ("You are a disciplined portfolio analyst for Indian equities. You explain and decide; you never "
          "calculate or recall figures. Every number you write must be copied from the FACTS you are given.")


class ReportState(TypedDict, total=False):
    holdings: list
    resolved: list
    regime: dict
    facts: dict
    portfolio: dict
    passages: dict
    flags: dict
    portfolio_flags: list
    previous: dict
    triage: dict
    investigations: list
    guardrails: list
    changes: list
    notes: list
    notes_attempt: int
    notes_problems: list
    llm_unavailable: bool
    summary_text: str
    summary_attempt: int
    summary_problems: list
    warnings: list
    model_used: list
    report_md: str
    figures: dict


@dataclass
class Toolbox:
    """Everything the graph needs from the outside world, injectable so tests need no network."""
    resolve: Callable[[str], dict]
    snapshot: Callable[[str], dict]
    benchmark: Callable[[str], dict]
    news: Callable[[str], dict]
    regime: Callable[[], dict]
    sector_of: Callable[[str], str]
    fundamentals: Callable = None      # (symbol) -> get_fundamentals-style dict (used by the research agent)
    risk: Callable = None              # (symbol) -> get_risk-style dict


def default_toolbox():
    """The real tools. Calls go through the smolagents Tool objects so the ledger records them."""
    from agent.tools import ALL_TOOLS
    from tools import resolver
    from tools.sector_map import lookup_sector
    t = {x.name: x for x in ALL_TOOLS}
    return Toolbox(
        resolve=resolver.resolve_stock,
        snapshot=lambda sym: t["get_stock_snapshot"](ticker=sym),
        benchmark=lambda sector: t["get_sector_benchmark"](sector=sector),
        news=lambda sym: t["get_stock_news"](ticker_or_name=sym),
        regime=lambda: t["get_market_regime"](),
        sector_of=lambda sym: lookup_sector(sym)["sector"],
        fundamentals=lambda sym: t["get_fundamentals"](ticker=sym),
        risk=lambda sym: t["get_risk"](ticker=sym),
    )


# --------------------------------------------------------------------------- helpers

def _num(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _parse_notes(text):
    """The model's JSON -> list of dicts. Tolerates code fences and a bare list."""
    if not text:
        return []
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start = min([i for i in (body.find("{"), body.find("[")) if i >= 0], default=-1)
    if start < 0:
        return []
    try:
        data = json.loads(body[start:body.rfind("}" if body[start] == "{" else "]") + 1])
    except ValueError:
        return []
    items = data.get("notes", []) if isinstance(data, dict) else data
    return [i for i in items if isinstance(i, dict)]


def _clean_note(item):
    action = str(item.get("action", "")).upper().strip()
    return {
        "symbol": str(item.get("symbol", "")).upper().strip(),
        "action": action if action in ACTIONS else "REVIEW",
        "rationale": str(item.get("rationale", "")).strip(),
        "trigger": str(item.get("trigger", "")).strip(),
    }


_FIELD_LABELS = [
    ("current_price", "Share price", "₹{}"), ("1m_change_pct", "1-month price change", "{}%"),
    ("3m_change_pct", "3-month price change", "{}%"), ("pct_from_52w_high", "Distance from 52-week high", "{}%"),
    ("pe_ratio", "P/E ratio", "{}"), ("roce_pct", "ROCE", "{}%"), ("roce_trend", "ROCE trend", "{}"),
    ("debt_to_equity", "Debt-to-equity ratio", "{}"), ("revenue_growth_pct", "Revenue growth", "{}%"),
    ("fcf_cr", "Free cash flow", "₹{} crore"), ("beta", "Beta against NIFTY 50", "{}"),
    ("volatility_pct", "Annualised volatility", "{}%"),
]
_FIELD_NAME = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")      # snake_case tokens such as pe_ratio


def _facts_text(state):
    """The facts as plain labelled lines. The model copies what it sees, so it must not see field names."""
    lines = []
    regime = state.get("regime", {})
    nifty = regime.get("nifty_returns_pct") or {}
    lines.append(f"MARKET: regime {regime.get('regime')}; NIFTY 50 returned {nifty.get('1w')}% over 1 week, "
                 f"{nifty.get('1m')}% over 1 month and {nifty.get('3m')}% over 3 months.")
    for r in state.get("resolved", []):
        f = state.get("facts", {}).get(r["symbol"])
        if not f:
            continue
        s = f["snapshot"]
        lines.append("")
        w = f.get("weight_pct")
        lines.append(f"{r['symbol']} ({r['company']}), sector {f['sector']}" + (f", {w}% of the portfolio" if w is not None else ""))
        for key, label, fmt in _FIELD_LABELS:
            v = s.get(key)
            if v is None:
                continue
            lines.append(f"  - {label}: " + (v if isinstance(v, str) else fmt.format(v)))
        if f.get("benchmark_name") and f.get("sector_1m") is not None:
            lines.append(f"  - Sector benchmark ({f['benchmark_name']}), 1-month change: {f['sector_1m']}%")
        if f.get("vs_sector_1m") is not None:
            ahead = "ahead of" if f["vs_sector_1m"] >= 0 else "behind"
            lines.append(f"  - Versus its sector over 1 month: {abs(f['vs_sector_1m'])} percentage points {ahead} the benchmark")
        for h in f.get("headlines", []):
            lines.append(f"  - Headline: {h}")
        for fl in state.get("flags", {}).get(r["symbol"], []):
            lines.append(f"  - Rule flag ({fl['kind']}, {fl['severity']}): {fl['title']}. {fl['detail']}")
        last = (state.get("previous") or {}).get("stocks", {}).get(r["symbol"])
        if last:
            lines.append(f"  - Last report's decision for this stock: {last.get('action')}")
        for invs in (i for i in state.get("investigations", []) if i["symbol"] == r["symbol"]):
            lines.append(f"  - Research question: {invs['question']}")
            for fd in invs["findings"]:
                cite = ("[" + ", ".join(fd["evidence"]) + "]") if fd["evidence"] else "[no evidence]"
                lines.append(f"  - Research finding {cite}: {fd['claim']}")
            for e in invs["evidence"]:
                lines.append(f"  - Evidence [{e['id']}] ({e['tool']}{', ' + e['source'] if e.get('source') else ''}): {e['text'][:350]}")
    for pf in state.get("portfolio_flags", []):
        lines.append("")
        lines.append(f"PORTFOLIO FLAG ({pf['severity']}): {pf['title']}. {pf['detail']}")
    return "\n".join(lines)


def _notes_prompt(state, problems=None):
    fix = ""
    if problems:
        fix = ("\nYOUR PREVIOUS ANSWER HAD PROBLEMS: " + "; ".join(problems[:8]) +
               ". Rewrite it: copy figures exactly as written in FACTS, write plain English, and do not use field names.\n")
    return f"""FACTS (the only source of figures you may use):
{_facts_text(state)}
{fix}
For EACH holding give a decision. Reply with JSON only, in this exact shape:
{{"notes": [{{"symbol": "TCS", "action": "HOLD", "rationale": "...", "trigger": "..."}}]}}

Rules:
- action is one of HOLD, ADD, REDUCE, EXIT, REVIEW (REVIEW when the facts are too thin to decide).
- rationale: 1 or 2 plain-English sentences, at most 40 words. Copy figures exactly as shown in FACTS.
  Say how the stock compares with its sector benchmark when that is given.
- trigger: ONE full sentence beginning "Reconsider if" naming a CHANGE from the current state in FACTS that would
  change the decision. If FACTS say the ROCE trend is stable, "turns stable" is not a change; say what it would have
  to turn into. Do NOT invent numeric thresholds.
- Use the stock's own weakest and strongest fact in the rationale (for example a big gap to its sector), not
  generic praise.
- Write every metric by its plain name exactly as labelled in FACTS ("P/E ratio", "1-month price change"). Never use underscores or code-style names.
- If a metric is missing or says "not applicable", say that plainly; do not guess it.
- Rule flags are prompts to look closer, not instructions. Say whether the flags and research change your decision.
- When you use a research finding or evidence, cite it as [E1]. Do not state anything the research does not show.
- A finding marked "[no evidence]" or saying the documents do not explain something means the cause is unknown: say so."""


def _summary_prompt(state, problems=None):
    p = state["portfolio"]
    c = p["concentration"]
    regime = state.get("regime", {})
    alloc = ", ".join(f"{k} {v}%" for k, v in p["sector_allocation"].items())
    decisions = ", ".join(f"{n['symbol']} {n['action']}" for n in state.get("notes", []))
    facts = f"""Market regime: {regime.get('regime')}
Total portfolio value: ₹{p['total_value']}
Overall return on holdings with a known cost: {p['overall_return_pct']}%
Sector weights: {alloc}
Largest sector: {c['largest_sector']} at {c['largest_sector_pct']}% of the portfolio
Largest holding: {c['top_holding_name']} at {c['top_holding_pct']}% of the portfolio
Portfolio beta (value-weighted): {p['beta_value_weighted']}
Decisions: {decisions}
Data notes: {'; '.join(p['data_quality']) or 'none'}
Portfolio flags: {'; '.join(f['title'] + ': ' + f['detail'] for f in state.get('portfolio_flags', [])) or 'none'}
{mem.summarize_changes(state.get('changes', []), (state.get('previous') or {}).get('date', '')) or 'Changes since last report: none (or first report)'}"""
    fix = ""
    if problems:
        fix = ("\nYOUR PREVIOUS ANSWER HAD PROBLEMS: " + "; ".join(problems[:8]) +
               ". Copy figures exactly as shown in FACTS and do not use field names.\n")
    return f"""FACTS:
{facts}
{fix}
Write a portfolio summary in plain English, at most 90 words: overall position, the main concentration risk, and
ONE specific thing to watch next week: name a holding, a sector or a risk (never "the decisions" or "performance").
Copy figures exactly as shown in FACTS. No bullet points, no headings, no field names."""


# --------------------------------------------------------------------------- the graph

def _search(kb, query, symbol):
    """Works with the LlamaIndex DocumentIndex (company-scoped) and the simple keyword KnowledgeBase."""
    try:
        return kb.search(query, k=3, symbol=symbol)
    except TypeError:
        return kb.search(query, k=3)


def build_graph(toolbox=None, llm=None, kb=None, today=None, memory=None, investigate=True,
                max_stocks=2, max_tool_calls=2):
    """Compile the report graph.

    llm=None gives a data-only report (no model calls). `kb` is a document index (see pipeline/rag.py),
    `memory` a pipeline.memory.Memory. The research agent makes at most 1 + max_stocks*(max_tool_calls+1)
    model requests; investigate=False skips it.
    """
    tb = toolbox or default_toolbox()
    day = today or date.today()
    stamp = day.isoformat()

    def resolve(state):
        ledger.reset()                      # the verifier must only see THIS run's facts
        resolved, warnings = [], list(state.get("warnings", []))
        for h in state["holdings"]:
            r = tb.resolve(h["name"])
            if r["status"] == "resolved":
                resolved.append({"name": h["name"], "symbol": r["symbol"], "company": r["name"],
                                 "quantity": h.get("quantity"), "avg_price": h.get("avg_price"),
                                 "note": r.get("note", "")})
                if r["confidence"] != "high":
                    warnings.append(f"'{h['name']}' was matched to {r['symbol']} ({r['name']}); please confirm.")
            elif r["status"] == "ambiguous":
                opts = "; ".join(f"{c['symbol']} ({c['name']})" for c in r["candidates"])
                warnings.append(f"'{h['name']}' skipped: it matches several companies ({opts}).")
            else:
                warnings.append(f"'{h['name']}' skipped: not found in the NSE equity list.")
        return {"resolved": resolved, "warnings": warnings}

    def market(state):
        regime = tb.regime()
        warnings = list(state.get("warnings", []))
        if not regime.get("ok", True):
            warnings.append("Market data unavailable: the regime could not be determined.")
        return {"regime": regime, "warnings": warnings}

    def stock_data(state):
        facts, warnings = {}, list(state.get("warnings", []))
        bench_cache = {}
        for r in state["resolved"]:
            snap = tb.snapshot(r["symbol"])
            if not snap.get("ok"):
                warnings.append(f"{r['symbol']}: no market data ({snap.get('message', 'unavailable')}).")
                continue
            sector = tb.sector_of(r["symbol"])
            if sector not in bench_cache:
                bench_cache[sector] = tb.benchmark(sector)
            b = bench_cache[sector]
            news = tb.news(r["symbol"])
            s = snap["snapshot"]
            sector_1m = (b.get("returns") or {}).get("1m") if b.get("ok") else None
            stock_1m = _num(s.get("1m_change_pct"))
            vs = round(stock_1m - sector_1m, 2) if stock_1m is not None and sector_1m is not None else None
            if vs is not None:
                ledger.record("computed", {"symbol": r["symbol"]}, {"symbol": r["symbol"], "vs_sector_1m_pct": vs})
            facts[r["symbol"]] = {
                "snapshot": s, "sector": sector, "company": r["company"],
                "benchmark_name": b.get("name") if b.get("ok") else None,
                "benchmark_kind": b.get("kind"), "sector_1m": sector_1m, "vs_sector_1m": vs,
                "headlines": [h["title"] + f" ({h['date']})" for h in (news.get("headlines") or [])[:3]],
                "caveats": snap.get("caveats", ""),
            }
        return {"facts": facts, "warnings": warnings}

    def portfolio(state):
        rows = []
        for r in state["resolved"]:
            f = state["facts"].get(r["symbol"])
            if not f:
                continue
            s, qty, avg = f["snapshot"], _num(r.get("quantity")), _num(r.get("avg_price"))
            price = _num(s.get("current_price"))
            value = price * qty if price is not None and qty and qty > 0 else None
            rows.append({
                "stock_name": r["company"], "ticker": r["symbol"], "sector": f["sector"], "quantity": qty,
                "current_value": value,
                "total_investment": avg * qty if avg and avg > 0 and qty and qty > 0 else None,
                "beta": _num(s.get("beta")), "trend_label": s.get("trend_label"),
                "volatility_bucket": volatility_bucket(_num(s.get("volatility_pct"))),
                "1m_change_pct": _num(s.get("1m_change_pct")),
            })
        df = pd.DataFrame(rows, columns=["stock_name", "ticker", "sector", "quantity", "current_value", "total_investment",
                                         "beta", "trend_label", "volatility_bucket", "1m_change_pct"])
        summary = summarize_portfolio(df)
        facts = {s: dict(f) for s, f in state["facts"].items()}
        for h in summary["top_holdings"]:
            if h["ticker"] in facts:
                facts[h["ticker"]]["weight_pct"] = h["weight_pct"]
        for sym, f in facts.items():                     # weights beyond the top 5
            if "weight_pct" not in f:
                row = df[df.ticker == sym]
                v = row["current_value"].iloc[0] if len(row) else None
                if v is not None and v == v and summary["total_value"]:
                    f["weight_pct"] = round(float(v) / summary["total_value"] * 100, 2)
        ledger_view = dict(summary)
        ledger_view["sector_allocation"] = [{"sector": k, "weight_pct": v} for k, v in summary["sector_allocation"].items()]
        ledger.record("summarize_portfolio", {}, ledger_view)
        return {"portfolio": summary, "facts": facts}

    def rules(state):
        flags = {}
        n_holdings = len(state.get("facts", {}))
        for sym, f in state.get("facts", {}).items():
            fl = rl.evaluate_stock(sym, f["snapshot"], weight_pct=f.get("weight_pct"),
                                   vs_sector_1m=f.get("vs_sector_1m"), n_holdings=n_holdings)
            flags[sym] = [x.as_dict() for x in fl]
            ledger.record("rules", {"symbol": sym}, {"symbol": sym, "flags": flags[sym]})
        pf = [x.as_dict() for x in rl.evaluate_portfolio(state["portfolio"])]
        ledger.record("rules", {"scope": "portfolio"}, {"flags": pf})
        return {"flags": flags, "portfolio_flags": pf}

    def memory_load(state):
        return {"previous": memory.previous(day) if memory is not None else None}

    def investigate_node(state):
        if llm is None or not investigate or not state.get("flags"):
            return {"investigations": [], "triage": {"how": "skipped", "picks": []}}
        flags = {s: [rl.Flag(**d) for d in lst] for s, lst in state["flags"].items()}
        companies = {r["symbol"]: r["company"] for r in state["resolved"]}
        tools = inv.Tools(docs=(lambda q, sym: _search(kb, q, sym)) if kb is not None and len(kb) else None,
                          news=lambda sym: tb.news(sym), fundamentals=tb.fundamentals, risk=tb.risk)
        sources = {}
        if kb is not None and len(kb):
            for sym in companies:
                has = getattr(kb, "covers", lambda s: True)(sym)
                sources[sym] = list(getattr(kb, "sources", lambda s: ["documents"])(sym)) if has else []
        has_docs = {s for s, v in sources.items() if v}
        picks, _, how = inv.triage(llm, flags, companies, max_stocks=max_stocks, has_docs=has_docs)
        results, warnings = [], list(state.get("warnings", []))
        for pick in picks:
            sym = pick["symbol"]
            r = inv.investigate_stock(llm, tools, sym, companies.get(sym, sym), state["facts"][sym]["sector"],
                                      pick["question"], flags[sym], max_tool_calls=max_tool_calls,
                                      doc_sources=sources.get(sym))
            results.append(r)
            if r["stopped"] == "model_unavailable":
                warnings.append(f"Research on {sym} stopped early: no model answered.")
        findings = [f for r in results for f in r["findings"]]
        figs = dict(state.get("figures", {}))
        figs["research"] = {"checked": len(findings), "supported": sum(f["verified"] for f in findings),
                            "problems": sum(not f["verified"] for f in findings)}
        used = list(state.get("model_used", []))
        used += [f"{a['provider']}/{a['model']}" for a in getattr(llm, "attempts", []) if a.get("ok")]
        return {"investigations": results, "triage": {"how": how, "picks": picks}, "figures": figs,
                "warnings": warnings, "model_used": used}

    def guardrails(state):
        flags = {s: [rl.Flag(**d) for d in lst] for s, lst in state.get("flags", {}).items()}
        notes, applied = [], []
        for n in state.get("notes", []):
            action, why = rl.apply_guardrails(n["action"], flags.get(n["symbol"], []))
            if why:
                applied.append({"symbol": n["symbol"], "from": n["action"], "to": action, "reason": why})
                n = {**n, "model_action": n["action"], "action": action, "guardrail": why}
            notes.append(n)
        return {"notes": notes, "guardrails": applied}

    def compare(state):
        return {"changes": mem.diff(state.get("previous"), mem.build_record(state))}

    def save_memory(state):
        if memory is not None and state.get("notes"):
            memory.save(mem.build_record(state), today=day)
        return {}

    def _call_llm(prompt, state):
        text, meta = llm.generate(prompt, system=SYSTEM)
        used = list(state.get("model_used", []))
        used.append(f"{meta['provider']}/{meta['model']}")
        return text, used

    def write_notes(state):
        if llm is None:
            return {"notes": [], "llm_unavailable": True, "notes_problems": [],
                    "warnings": state.get("warnings", []) + ["Analysis not generated (running without a model)."]}
        attempt = state.get("notes_attempt", 0) + 1
        try:
            text, used = _call_llm(_notes_prompt(state, state.get("notes_problems") if attempt > 1 else None), state)
        except AllModelsFailed as e:
            return {"notes": [], "llm_unavailable": True, "notes_problems": [], "notes_attempt": attempt,
                    "warnings": state.get("warnings", []) + [f"Analysis not generated: no model answered ({e})."]}
        parsed = [_clean_note(i) for i in _parse_notes(text)]
        by_symbol = {n["symbol"]: n for n in parsed}
        notes = []
        for r in state["resolved"]:
            if r["symbol"] not in state.get("facts", {}):
                continue
            notes.append(by_symbol.get(r["symbol"]) or {"symbol": r["symbol"], "action": "REVIEW",
                                                         "rationale": "The model returned no note for this holding.", "trigger": ""})
        return {"notes": notes, "notes_attempt": attempt, "model_used": used}

    def check_notes(state):
        facts_ledger = ledger.snapshot()
        problems, checked, supported = [], 0, 0
        notes = []
        for n in state.get("notes", []):
            rep = verify_answer(f"{n['rationale']} {n['trigger']}", facts_ledger)
            checked += len(rep["claims"])
            supported += rep["n_supported"] + rep["n_derived"]
            found = [f"{n['symbol']}: {c['claim']} is not supported by the facts" for c in rep["problems"]]
            names = _FIELD_NAME.findall(f"{n['rationale']} {n['trigger']}")
            if names:
                found.append(f"{n['symbol']}: uses internal field names ({', '.join(names[:3])}); write plain English")
            if n["action"] != "REVIEW" and len(n["trigger"].split()) < 4:
                found.append(f"{n['symbol']}: the trigger is not a full sentence condition")
            problems += found
            notes.append({**n, "verified": rep["n_problems"] == 0 and not names and not (len(n["trigger"].split()) < 4 and n["action"] != "REVIEW"),
                          "report": rep})
        final = (not problems) or state.get("notes_attempt", 1) >= MAX_ATTEMPTS
        if final:
            for n in notes:
                for key in ("rationale", "trigger"):
                    n[key] = annotate(n[key], n["report"]) if n["report"]["problems"] else n[key]
        figs = dict(state.get("figures", {}))
        figs["notes"] = {"checked": checked, "supported": supported, "problems": len(problems)}
        return {"notes": notes, "notes_problems": problems, "figures": figs}

    def route_notes(state):
        if state.get("llm_unavailable") or not state.get("notes_problems"):
            return "guardrails"
        return "guardrails" if state.get("notes_attempt", 1) >= MAX_ATTEMPTS else "write_notes"

    def write_summary(state):
        if llm is None or state.get("llm_unavailable"):
            return {"summary_text": "", "summary_problems": []}
        attempt = state.get("summary_attempt", 0) + 1
        try:
            text, used = _call_llm(_summary_prompt(state, state.get("summary_problems") if attempt > 1 else None), state)
        except AllModelsFailed as e:
            return {"summary_text": "", "summary_problems": [], "summary_attempt": attempt, "llm_unavailable": True,
                    "warnings": state.get("warnings", []) + [f"Summary not generated: no model answered ({e})."]}
        return {"summary_text": text.strip(), "summary_attempt": attempt, "model_used": used}

    def check_summary(state):
        text = state.get("summary_text", "")
        if not text:
            return {"summary_problems": []}
        rep = verify_answer(text, ledger.snapshot())
        problems = [f"{c['claim']} is not supported by the facts" for c in rep["problems"]]
        names = _FIELD_NAME.findall(text)
        if names:
            problems.append(f"uses internal field names ({', '.join(names[:3])}); write plain English")
        figs = dict(state.get("figures", {}))
        prev = figs.get("summary", {"checked": 0, "supported": 0})
        figs["summary"] = {"checked": len(rep["claims"]), "supported": rep["n_supported"] + rep["n_derived"],
                           "problems": len(problems)}
        out = {"summary_problems": problems, "figures": figs}
        if not problems or state.get("summary_attempt", 1) >= MAX_ATTEMPTS:
            out["summary_text"] = annotate(text, rep) if problems else text
        return out

    def route_summary(state):
        if state.get("llm_unavailable") or not state.get("summary_problems"):
            return "save_memory"
        return "save_memory" if state.get("summary_attempt", 1) >= MAX_ATTEMPTS else "write_summary"

    def assemble(state):
        return {"report_md": render_report(state, stamp)}

    g = StateGraph(ReportState)
    for name, fn in [("resolve", resolve), ("market", market), ("stock_data", stock_data), ("portfolio", portfolio),
                     ("rules", rules), ("memory_load", memory_load), ("investigate", investigate_node),
                     ("write_notes", write_notes), ("check_notes", check_notes), ("guardrails", guardrails),
                     ("compare", compare), ("write_summary", write_summary), ("check_summary", check_summary),
                     ("save_memory", save_memory), ("assemble", assemble)]:
        g.add_node(name, fn)
    g.add_edge(START, "resolve")
    g.add_edge("resolve", "market")
    g.add_edge("market", "stock_data")
    g.add_edge("stock_data", "portfolio")
    g.add_edge("portfolio", "rules")
    g.add_edge("rules", "memory_load")
    g.add_edge("memory_load", "investigate")
    g.add_edge("investigate", "write_notes")
    g.add_edge("write_notes", "check_notes")
    g.add_conditional_edges("check_notes", route_notes, {"write_notes": "write_notes", "guardrails": "guardrails"})
    g.add_edge("guardrails", "compare")
    g.add_edge("compare", "write_summary")
    g.add_edge("write_summary", "check_summary")
    g.add_conditional_edges("check_summary", route_summary, {"write_summary": "write_summary", "save_memory": "save_memory"})
    g.add_edge("save_memory", "assemble")
    g.add_edge("assemble", END)
    return g.compile()


# --------------------------------------------------------------------------- the report

def _fmt(v, nd=2, suffix=""):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    if isinstance(v, str):
        return v
    return f"{v:,.{nd}f}{suffix}"


def render_report(state, stamp):
    regime = state.get("regime", {})
    nifty = regime.get("nifty_returns_pct") or {}
    p = state.get("portfolio", {})
    L = [f"# Weekly portfolio report: {stamp}", ""]
    L += ["> Decision support, not investment advice. Every figure below was fetched by code; figures written by "
          "the model were checked against those results and are tagged [unverified] if they could not be.", ""]

    L += ["## Market", "",
          f"**Regime: {regime.get('regime', 'unknown')}**. NIFTY 50: 1 week {_fmt(nifty.get('1w'), 2, '%')}, "
          f"1 month {_fmt(nifty.get('1m'), 2, '%')}, 3 months {_fmt(nifty.get('3m'), 2, '%')}. "
          f"_The regime is a simple rule on NIFTY's 1- and 3-month returns, not a forecast._", ""]

    if state.get("previous"):
        L += [f"## What changed since {state['previous'].get('date')}", ""]
        L += [f"- {c['detail']}" for c in state.get("changes", [])] or ["- No material changes."]
        L.append("")

    L += ["## Portfolio", ""]
    if p.get("holdings"):
        c = p["concentration"]
        L += [f"- Value of valued holdings: ₹{_fmt(p['total_value'], 0)} across {p['holdings']} holdings",
              f"- Return, like-for-like (holdings with a known cost): {_fmt(p['overall_return_pct'], 2, '%')}",
              f"- Largest holding: {c['top_holding_name']} ({_fmt(c['top_holding_pct'], 1, '%')}); top 5 = {_fmt(c['top5_pct'], 1, '%')}",
              f"- Largest sector: {c['largest_sector']} ({_fmt(c['largest_sector_pct'], 1, '%')})",
              f"- Beta (value-weighted): {_fmt(p['beta_value_weighted'])}"]
        L += ["", "| Sector | Weight |", "|---|---|"]
        L += [f"| {s} | {w:.1f}% |" for s, w in p["sector_allocation"].items()]
        for note in p.get("data_quality", []):
            L.append(f"\n> Data note: {note}")
        for pf in state.get("portfolio_flags", []):
            L.append(f"\n> ⚑ Portfolio flag ({pf['severity']}): {pf['title']}. {pf['detail']}")
    else:
        L.append("No holdings could be valued.")
    if state.get("summary_text"):
        L += ["", f"**Summary:** {state['summary_text']}"]

    L += ["", "## Holdings", "", "| Stock | Weight | Price | 1M | vs sector (1M) | P/E | ROCE (trend) | D/E |",
          "|---|---|---|---|---|---|---|---|"]
    for r in state.get("resolved", []):
        f = state.get("facts", {}).get(r["symbol"])
        if not f:
            L.append(f"| {r['symbol']} | n/a | no data | | | | | |")
            continue
        s = f["snapshot"]
        roce = _fmt(s.get("roce_pct"), 1, "%")
        trend = s.get("roce_trend")
        roce_cell = roce if not isinstance(s.get("roce_pct"), str) else "n/a (financial)"
        if trend and not str(trend).startswith("not applicable"):
            roce_cell += f" ({trend})"
        de = s.get("debt_to_equity")
        L.append(f"| {r['symbol']} | {_fmt(f.get('weight_pct'), 1, '%')} | ₹{_fmt(s.get('current_price'), 1)} | "
                 f"{_fmt(s.get('1m_change_pct'), 1, '%')} | {_fmt(f.get('vs_sector_1m'), 1, ' pp')} | "
                 f"{_fmt(s.get('pe_ratio'), 1)} | {roce_cell} | {'n/a' if isinstance(de, str) else _fmt(de)} |")

    L += ["", "## Decisions", ""]
    notes = state.get("notes", [])
    if not notes:
        L.append("_Analysis was not generated for this run._")
    for n in notes:
        f = state.get("facts", {}).get(n["symbol"], {})
        mark = "✓ verified" if n.get("verified") else "⚠ contains unverified figures"
        L += [f"### {n['symbol']} — {n['action']}  ({mark})", "", n["rationale"] or "_no rationale_"]
        if n["trigger"]:
            L += ["", f"*Would change if:* {n['trigger']}"]
        fl = state.get("flags", {}).get(n["symbol"], [])
        if fl:
            L += ["", "*Rule flags:* " + "; ".join(f"{'⚠' if x['kind'] == 'concern' else '✓'} {x['title']}" for x in fl)]
        if n.get("guardrail"):
            L += ["", f"_Guardrail: {n['guardrail']}_"]
        if f.get("benchmark_name"):
            L += ["", f"_Sector benchmark: {f['benchmark_name']}_"]
        for h in f.get("headlines", []):
            L.append(f"- {h}")
        if f.get("headlines"):
            L.append("")
        L.append("")

    if state.get("triage"):
        t = state["triage"]
        L += ["## Research agent", "",
              f"Chosen to investigate ({t.get('how')}): " + (", ".join(p['symbol'] for p in t.get("picks", [])) or "nothing"), ""]
        for i in state.get("investigations", []):
            L += [f"### {i['symbol']}: {i['question']}", ""]
            for st in i["steps"]:
                arg = ", ".join(f"{k}={v!r}" for k, v in st["args"].items())
                L.append(f"{st['n']}. **{st['action']}**({arg})" + (f": {st['why']}" if st["why"] else "")
                         + (f" → {', '.join(st['evidence'])}" if st["evidence"] else ""))
            L.append("")
            for fd in i["findings"]:
                mark = "✓" if fd["verified"] else "⚠ " + "; ".join(fd["issues"]) + ":"
                cite = f" [{', '.join(fd['evidence'])}]" if fd["evidence"] else ""
                L.append(f"- {mark} {fd['claim']}{cite}")
            for e in i["evidence"]:
                if e["source"]:
                    L.append(f"  > [{e['id']}] {e['source']}, p.{e['page']}: {e['text'][:260]}…")
            L.append("")

    figs = state.get("figures", {})
    checked = sum(v.get("checked", 0) for v in figs.values())
    supported = sum(v.get("supported", 0) for v in figs.values())
    problems = sum(v.get("problems", 0) for v in figs.values())
    L += ["## Verification", ""]
    if figs:
        L.append(f"- {supported} of {checked} checked figures/claims traced to a tool result; {problems} unsupported.")
    L.append(f"- Model(s): {', '.join(dict.fromkeys(state.get('model_used', []))) or 'none (data-only run)'}")
    L.append(f"- Model-written sections attempted: notes x{state.get('notes_attempt', 0)}, summary x{state.get('summary_attempt', 0)}")
    for w in state.get("warnings", []):
        L.append(f"- Warning: {w}")
    return "\n".join(L).rstrip() + "\n"
