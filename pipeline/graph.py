"""The weekly report as a LangGraph graph.

    resolve -> market -> stock_data -> portfolio -> documents -> write_notes -> check_notes
                                                                    ^              |
                                                                    +---retry------+
                                                       -> write_summary -> check_summary -> assemble

Code gathers every fact (no model involved) and records it in the ledger. The model is used at exactly
two nodes, write_notes and write_summary, and each result goes through the verifier. When a note
contains a figure no tool returned, the node is re-run once with the problems listed; whatever is
still unsupported is tagged [unverified] in the report instead of being passed off as fact.
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


def _facts_for_prompt(state):
    out = []
    for r in state.get("resolved", []):
        f = state.get("facts", {}).get(r["symbol"])
        if not f:
            continue
        s = f["snapshot"]
        keys = ("current_price", "1m_change_pct", "3m_change_pct", "pct_from_52w_high", "pe_ratio", "roce_pct",
                "roce_trend", "debt_to_equity", "revenue_growth_pct", "fcf_cr", "beta", "volatility_pct")
        entry = {
            "symbol": r["symbol"], "company": r["company"], "sector": f["sector"],
            "weight_pct": f.get("weight_pct"),
            "metrics": {k: s.get(k) for k in keys if s.get(k) is not None},
            "sector_benchmark": f.get("benchmark_name"),
            "sector_1m_change_pct": f.get("sector_1m"),
            "vs_sector_1m_pct": f.get("vs_sector_1m"),
            "headlines": f.get("headlines", []),
            "documents": [{"id": p["id"], "source": p["source"], "text": p["text"][:400]}
                          for p in state.get("passages", {}).get(r["symbol"], [])],
        }
        out.append(entry)
    return out


def _notes_prompt(state, problems=None):
    facts = json.dumps({"market_regime": state.get("regime", {}).get("regime"),
                        "nifty_returns_pct": state.get("regime", {}).get("nifty_returns_pct"),
                        "holdings": _facts_for_prompt(state)}, ensure_ascii=False)
    fix = ""
    if problems:
        fix = ("\nYOUR PREVIOUS ANSWER HAD FIGURES THAT ARE NOT IN THE FACTS: " + ", ".join(problems[:8]) +
               ". Rewrite using only figures that appear in FACTS; drop any figure you cannot find there.\n")
    return f"""FACTS (the only source of figures you may use):
{facts}
{fix}
For EACH holding give a decision. Reply with JSON only, in this exact shape:
{{"notes": [{{"symbol": "TCS", "action": "HOLD", "rationale": "...", "trigger": "..."}}]}}

Rules:
- action is one of HOLD, ADD, REDUCE, EXIT, REVIEW (use REVIEW when the facts are too thin to decide).
- rationale: at most 40 words, using only figures present in FACTS. Mention how the stock compares with its sector benchmark when given.
- trigger: one condition that would change the decision, stated as a trend, event or a figure already in FACTS.
  Do NOT invent numeric thresholds.
- If a metric is missing or says "not applicable", say so; do not guess it.
- Cite a document as [D1] only if you used it."""


def _summary_prompt(state, problems=None):
    p = state["portfolio"]
    facts = json.dumps({
        "market_regime": state.get("regime", {}).get("regime"),
        "nifty_returns_pct": state.get("regime", {}).get("nifty_returns_pct"),
        "total_value": p["total_value"], "overall_return_pct": p["overall_return_pct"],
        "sector_allocation_pct": p["sector_allocation"], "largest_sector": p["concentration"]["largest_sector"],
        "largest_sector_pct": p["concentration"]["largest_sector_pct"],
        "top_holding": p["concentration"]["top_holding_name"], "top_holding_pct": p["concentration"]["top_holding_pct"],
        "beta_value_weighted": p["beta_value_weighted"], "data_quality": p["data_quality"],
        "decisions": [{"symbol": n["symbol"], "action": n["action"]} for n in state.get("notes", [])],
    }, ensure_ascii=False)
    fix = ""
    if problems:
        fix = ("\nYOUR PREVIOUS ANSWER HAD FIGURES NOT IN THE FACTS: " + ", ".join(problems[:8]) +
               ". Use only figures that appear in FACTS.\n")
    return f"""FACTS:
{facts}
{fix}
Write a portfolio summary in plain text, at most 90 words: overall position, the main concentration risk,
and the one thing to watch next week. Use only figures from FACTS. No bullet points, no headings."""


# --------------------------------------------------------------------------- the graph

def build_graph(toolbox=None, llm=None, kb=None, today=None):
    """Compile the report graph. llm=None produces a data-only report (no model calls at all)."""
    tb = toolbox or default_toolbox()
    stamp = (today or date.today()).isoformat()

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

    def documents(state):
        passages = {}
        if kb is not None and len(kb):
            n = 0
            for r in state["resolved"]:
                hits = kb.search(f"{r['company']} {r['symbol']} outlook risks margins guidance", k=2)
                if hits:
                    items = []
                    for h in hits:
                        n += 1
                        items.append({"id": f"D{n}", "source": h["source"], "page": h.get("page"), "text": h["text"]})
                    passages[r["symbol"]] = items
                    ledger.record("search_documents", {"symbol": r["symbol"]}, {"symbol": r["symbol"], "passages": items})
        return {"passages": passages}

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
            problems += [f"{n['symbol']}: {c['claim']}" for c in rep["problems"]]
            notes.append({**n, "verified": rep["n_problems"] == 0, "report": rep})
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
            return "write_summary"
        return "write_summary" if state.get("notes_attempt", 1) >= MAX_ATTEMPTS else "write_notes"

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
        problems = [c["claim"] for c in rep["problems"]]
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
            return "assemble"
        return "assemble" if state.get("summary_attempt", 1) >= MAX_ATTEMPTS else "write_summary"

    def assemble(state):
        return {"report_md": render_report(state, stamp)}

    g = StateGraph(ReportState)
    for name, fn in [("resolve", resolve), ("market", market), ("stock_data", stock_data), ("portfolio", portfolio),
                     ("documents", documents), ("write_notes", write_notes), ("check_notes", check_notes),
                     ("write_summary", write_summary), ("check_summary", check_summary), ("assemble", assemble)]:
        g.add_node(name, fn)
    g.add_edge(START, "resolve")
    g.add_edge("resolve", "market")
    g.add_edge("market", "stock_data")
    g.add_edge("stock_data", "portfolio")
    g.add_edge("portfolio", "documents")
    g.add_edge("documents", "write_notes")
    g.add_edge("write_notes", "check_notes")
    g.add_conditional_edges("check_notes", route_notes, {"write_notes": "write_notes", "write_summary": "write_summary"})
    g.add_edge("write_summary", "check_summary")
    g.add_conditional_edges("check_summary", route_summary, {"write_summary": "write_summary", "assemble": "assemble"})
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
        if f.get("benchmark_name"):
            L += ["", f"_Sector benchmark: {f['benchmark_name']}_"]
        for h in f.get("headlines", []):
            L.append(f"- {h}")
        if f.get("headlines"):
            L.append("")
        for d in state.get("passages", {}).get(n["symbol"], []):
            L.append(f"> [{d['id']}] {d['source']}" + (f", p.{d['page']}" if d.get("page") else "") + f": {d['text'][:220]}…")
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
