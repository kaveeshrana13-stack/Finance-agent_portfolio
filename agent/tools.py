"""smolagents tools: thin wrappers over tools/. Each returns a plain dict.

A code-writing agent naturally indexes results (result["data"]["roce_pct"]). Returning JSON
strings made the model write code that crashed with "string indices must be integers", so the
tools return dicts of plain JSON-safe values.

The docstrings are what the model reads to decide which tool to call, so they say what each
tool answers and what its units are. Wrappers add no logic; every number comes from tools/.
"""
import json
import sys
from pathlib import Path

from smolagents import tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.ledger import record as _record
from tools import market as _market
from tools import news as _news
from tools import resolver as _resolver
from tools import stocks as _stocks


def _dump(obj):
    """Round-trip through JSON so the result holds only plain str/number/bool/list/dict values."""
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


_MATCH = {}      # symbol -> how the user's text was matched; attached to every stock result


def _dump_matched(obj, symbol):
    """_dump(obj) plus 'matched': which company the input was resolved to, and how sure we are."""
    out = _dump(obj)
    if isinstance(out, dict) and symbol in _MATCH:
        out["matched"] = _MATCH[symbol]
    return out


def _sym(query):
    """Resolve a company name or symbol. Returns (symbol, None) or (None, error_dict).

    Done inside the tools so the model cannot skip it: an ambiguous name comes back as a
    ready-made message instead of a failed price lookup.
    """
    r = _resolver.resolve_stock(query)
    if r["status"] == "resolved":
        _MATCH[r["symbol"]] = {"query": query, "company": r["name"], "match": r["match_type"],
                               "confidence": r["confidence"], "note": r["note"]}
        return r["symbol"], None
    if r["status"] == "ambiguous":
        options = "; ".join(f"{c['symbol']} ({c['name']})" for c in r["candidates"])
        return None, {"ok": False, "status": "ambiguous", "candidates": r["candidates"],
                      "message": f"'{query}' is not a single listed company: {options}. Ask which one is meant."}
    return str(query).strip(), None          # not in the NSE list: let Yahoo try it as a bare symbol


def _brief(envelope, keys=("status", "symbol", "as_of", "data", "warnings")):
    out = {k: envelope[k] for k in keys if k in envelope}
    out["ok"] = envelope.get("status") in ("ok", "partial")      # True when the data is usable
    return out


@tool
def resolve_stock(query: str) -> dict:
    """Finds the official NSE symbol for a company name, nickname, symbol or ISIN.
    Use this first when the user gives a company name instead of a ticker.
    Returns a dict: ok (True only when resolved), status ('resolved'|'ambiguous'|'not_found'), symbol, name,
    candidates, note. When ok is False, message explains why: pass it on to the user and do not guess a symbol.
    Args:
        query: Company name or symbol, e.g. 'Zomato', 'Larsen & Toubro', 'TCS'.
    """
    r = _resolver.resolve_stock(query)
    out = {k: r[k] for k in ("status", "symbol", "name", "confidence", "candidates", "note")}
    out["ok"] = r["status"] == "resolved"
    if r["status"] == "ambiguous":
        options = "; ".join(f"{c['symbol']} ({c['name']})" for c in r["candidates"])
        out["message"] = f"'{query}' matches more than one listed company: {options}. Ask which one is meant."
    elif r["status"] == "not_found":
        out["message"] = f"'{query}' was not found in the NSE equity list."
    return _dump(out)


@tool
def get_stock_snapshot(ticker: str) -> dict:
    """One-call overview of an NSE stock: price and momentum, valuation, business quality and risk.
    Prefer this for any question that mixes several of price, P/E, ROCE, debt, growth, beta or volatility,
    and for comparing stocks (call it once per stock).
    Returns a dict: ok (True when data is usable), symbol, snapshot{current_price, 52w_high, 52w_low,
    pct_from_52w_high, 1m_change_pct, 3m_change_pct,
    trend_label, pe_ratio, pb_ratio, dividend_yield, market_cap_cr, roce_pct, roce_trend, roe, debt_to_equity,
    revenue_growth_pct, net_margin_pct, fcf_cr, beta, volatility_pct, max_drawdown_1y_pct}, caveats.
    Units: *_pct are percent, *_cr are INR crore, debt_to_equity is a plain ratio.
    The result's 'matched' field names the company the input was resolved to: state that company in your
    answer, and mention it if confidence is not 'high'.
    Args:
        ticker: NSE symbol or company name, e.g. 'TCS' or 'Zomato'.
    """
    ticker, err = _sym(ticker)
    if err:
        return _dump({**err, "symbol": None, "snapshot": {}})
    m = _stocks.fetch_stock_metrics_yf(ticker)
    if not m:
        return _dump({"ok": False, "symbol": ticker, "snapshot": {},
                      "message": f"No live data found for '{ticker}' on NSE or BSE."})
    keys = ("current_price", "52w_high", "52w_low", "pct_from_52w_high", "1m_change_pct", "3m_change_pct",
            "trend_label", "pe_ratio", "pb_ratio",
            "dividend_yield", "market_cap_cr", "roce_pct", "roce_trend", "roe", "debt_to_equity",
            "revenue_growth_pct", "net_margin_pct", "fcf_cr", "beta", "volatility_pct", "max_drawdown_1y_pct")
    return _dump_matched({"ok": True, "symbol": ticker, "snapshot": {k: m.get(k) for k in keys},
                          "fundamentals_fy": m.get("fundamentals_fy"), "caveats": m.get("data_warnings", "")}, ticker)


@tool
def get_price_momentum(ticker: str) -> dict:
    """Current price, 52-week range and 1-week/1-month/3-month price change (percent) of an NSE stock.
    Works for individual stocks only, NOT for indices such as NIFTY (use get_market_regime for NIFTY).
    Returns a dict: ok (True when data is usable), status ('ok'|'partial'|'error'), symbol,
    data{current_price, 1w_change_pct, 1m_change_pct, 3m_change_pct, ...}, warnings.
    Args:
        ticker: NSE symbol, e.g. 'TCS', 'BAJAJ-AUTO'.
    """
    ticker, err = _sym(ticker)
    if err:
        return _dump(err)
    return _dump_matched(_brief(_stocks.get_price_momentum(ticker)), ticker)


@tool
def get_valuation(ticker: str) -> dict:
    """Valuation of an NSE stock: P/E, P/B, PEG, market cap (INR crore) and dividend yield (percent).
    Returns a dict: ok (True when data is usable), status ('ok'|'partial'|'error'), symbol,
    data{pe_ratio, pb_ratio, peg_ratio, market_cap_cr, dividend_yield}, warnings.
    Args:
        ticker: NSE symbol, e.g. 'TITAN'.
    """
    ticker, err = _sym(ticker)
    if err:
        return _dump(err)
    return _dump_matched(_brief(_stocks.get_valuation(ticker)), ticker)


@tool
def get_fundamentals(ticker: str) -> dict:
    """Business quality of an NSE stock from annual statements: ROCE, ROE, EBIT margin, net margin,
    revenue growth, D/E ratio, free cash flow (INR crore), their trends, and a per-year history.
    Banks and financials return 'not_applicable' for ROCE, EBIT margin, FCF and D/E.
    Returns a dict: ok (True when data is usable), status ('ok'|'partial'|'error'), symbol,
    data{summary{roce_pct, roe_pct, de_ratio, fcf_cr, roce_trend ('improving'|'declining'|'stable'), margin_trend,
    fcf_trend, ...}, history_newest_first[...], not_applicable}, warnings.
    Args:
        ticker: NSE symbol, e.g. 'TITAN'.
    """
    ticker, err = _sym(ticker)
    if err:
        return _dump(err)
    r = _stocks.get_fundamentals(ticker)
    data = r["data"]
    keep = ("fy", "revenue_cr", "ebit_margin_pct", "roce_pct", "roe_pct", "de_ratio", "fcf_cr")
    compact = {
        "summary": data.get("summary", {}),
        "history_newest_first": [{k: h[k] for k in keep} for h in data.get("history", [])],
        "not_applicable": data.get("not_applicable", []),
    }
    return _dump_matched({**_brief(r, ("status", "symbol", "as_of", "warnings")), "data": compact}, ticker)


@tool
def get_risk(ticker: str) -> dict:
    """Risk of an NSE stock: annualized volatility (percent), beta against NIFTY 50, maximum drawdown
    over one year (percent) and how close the next earnings report is.
    Returns a dict: ok (True when data is usable), status ('ok'|'partial'|'error'), symbol,
    data{volatility_pct, beta, max_drawdown_1y_pct, earnings_proximity}, warnings.
    Args:
        ticker: NSE symbol, e.g. 'INDIGO'.
    """
    ticker, err = _sym(ticker)
    if err:
        return _dump(err)
    return _dump_matched(_brief(_stocks.get_risk(ticker)), ticker)


@tool
def get_market_regime() -> dict:
    """The current Indian market regime (Risk-On, Selective Risk-On, Range-bound, Risk-Off, Mixed)
    AND the NIFTY 50 index returns over 1 week, 1 month and 3 months (percent). Takes no arguments.
    Use this for any question about the market or NIFTY.
    Returns a dict: ok, regime, nifty_returns_pct{1w, 1m, 3m}, meaning, note.
    """
    returns = _market.fetch_index_returns("NIFTY 50")
    regime, description = _market.classify_market_regime(returns)
    return _dump({"ok": regime != "Unknown", "regime": regime, "nifty_returns_pct": returns,
                  "meaning": description.split("\n")[0],
                  "note": "The regime is a simple rule on NIFTY 1m/3m returns, not a forecast."})


@tool
def get_sector_benchmark(sector: str) -> dict:
    """Recent performance (percent over 1w/1m/3m) of a sector, to compare a stock against its sector.
    The result says which benchmark was used (index, ETF stand-in, basket) or that none exists.
    Returns a dict: ok (True when a benchmark exists), sector, kind ('index'|'etf'|'basket'|'none'), name,
    returns{1w, 1m, 3m}, note.
    Args:
        sector: Sector name such as 'IT', 'Banking', 'Pharma', 'FMCG', 'Auto', 'Energy', 'Metal',
            'Infrastructure', 'Financial Services', 'Defence', 'Chemicals', 'Utilities', 'Real Estate'.
    """
    b = _market.get_sector_benchmark(sector)
    return _dump({"ok": b["available"], **{k: b[k] for k in ("sector", "available", "kind", "name", "returns", "note")}})


@tool
def get_stock_news(ticker_or_name: str) -> dict:
    """Recent Moneycontrol headlines (last 7 days, titles only) that mention the company.
    An empty list means nothing was published, not that nothing happened.
    Returns a dict: ok (True when the fetch worked, even if count is 0), status, count, headlines[{date, title}],
    warnings.
    Args:
        ticker_or_name: NSE symbol or company name, e.g. 'INFY' or 'Infosys'.
    """
    r = _news.get_stock_news(ticker_or_name)
    items = [{"date": i["date"], "title": i["title"]} for i in r["data"]["items"]]
    return _dump({"ok": r["status"] in ("ok", "partial"), "status": r["status"], "source": r["source"], "count": len(items),
                  "headlines": items, "warnings": r["warnings"]})


ALL_TOOLS = [resolve_stock, get_stock_snapshot, get_price_momentum, get_valuation, get_fundamentals, get_risk,
             get_market_regime, get_sector_benchmark, get_stock_news]


# What the model sees. Nine similar tools made a 7B model pick badly (it skipped the overview tool and
# answered "ROCE: Not Available" without ever fetching it), so it gets the five it needs. The granular
# tools stay in ALL_TOOLS for graph nodes that call them deterministically.
AGENT_TOOLS = [get_stock_snapshot, get_market_regime, get_sector_benchmark, get_stock_news, resolve_stock]


def _attach_ledger(tools):
    """Make each tool record (name, args, full result) in the ledger whenever it is called.

    Done on the Tool objects rather than with an extra decorator, which smolagents warns about.
    """
    for t in tools:
        original = t.forward

        def forward(*args, __orig=original, __name=t.name, **kwargs):
            result = __orig(*args, **kwargs)
            _record(__name, kwargs or list(args), result)
            return result
        t.forward = forward


_attach_ledger(ALL_TOOLS)
