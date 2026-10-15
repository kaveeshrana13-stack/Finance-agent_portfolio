"""Stock data tools backed by Yahoo Finance (yfinance).

Four small tools, each answering one question, plus a backward-compatible composite:

    get_price_momentum(ticker)  How has the price moved?
    get_valuation(ticker)       Is it expensive?
    get_fundamentals(ticker)    Is the business good? (computed from annual statements)
    get_risk(ticker)            How risky is it? (volatility, beta vs NIFTY, drawdown)
    fetch_stock_metrics_yf      Flat dict of all of the above, for the Streamlit UI.

Every tool returns an envelope: {status, symbol, yahoo_symbol, as_of, source, data, warnings}.
Units are explicit: *_pct are percent numbers (2.08 means 2.08%), *_cr are INR crore,
*_ratio are plain ratios.

Known limits (also surfaced in `warnings`): yfinance is unofficial and gives about 4 annual
statements; banks/NBFCs/insurers report differently, so ROCE, EBIT margin, FCF and D/E are
marked not applicable for them instead of being computed.
"""
import math
import time
from datetime import datetime

import pandas as pd

from tools._yf import yf, YFINANCE_AVAILABLE
from tools.tickers import clean_ticker_symbol

CACHE_TTL_SECONDS = 600
CRORE = 1e7
TRADING_DAYS = 252

_cache = {}


def clear_cache():
    _cache.clear()


def _cached(key, loader):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]
    value = loader()
    if value is not None:
        _cache[key] = (now, value)
    return value


# --------------------------------------------------------------------------- data access

def _get_info(ysym):
    def load():
        try:
            return yf.Ticker(ysym).info or None
        except Exception:
            return None
    return _cached(("info", ysym), load)


def _get_history(ysym, period):
    def load():
        try:
            h = yf.Ticker(ysym).history(period=period)
            h = h.dropna(subset=["Close"])
            return h if len(h) >= 2 else None
        except Exception:
            return None
    return _cached(("hist", ysym, period), load)


def _get_statements(ysym):
    def load():
        try:
            t = yf.Ticker(ysym)
            return (t.financials, t.balance_sheet, t.cashflow)
        except Exception:
            return None
    return _cached(("stmts", ysym), load)


def _locate(ticker):
    """Find the Yahoo symbol (NSE first, then BSE) that has a live price.

    Returns (yahoo_symbol, exchange_suffix, info) or None.
    """
    sym = clean_ticker_symbol(ticker)
    if not sym or not YFINANCE_AVAILABLE or yf is None:
        return None
    for exchange in (".NS", ".BO"):
        info = _get_info(sym + exchange)
        if info and (info.get("currentPrice") or info.get("regularMarketPrice")):
            return sym + exchange, exchange, info
    return None


# --------------------------------------------------------------------------- small helpers

def _num(x):
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def _r(x, nd=2):
    v = _num(x)
    return None if v is None else round(v, nd)


def _pct(numer, denom, nd=2):
    n, d = _num(numer), _num(denom)
    if n is None or d is None or d == 0:
        return None
    return round(n / d * 100, nd)


def _envelope(symbol, ysym, source, data, warnings, status=None):
    return {
        "status": status or ("ok" if not warnings else "partial"),
        "symbol": symbol,
        "yahoo_symbol": ysym,
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "data": data,
        "warnings": warnings,
    }


def _not_found(ticker, source):
    return _envelope(clean_ticker_symbol(ticker), None, source, {},
                     [f"No live price found for '{ticker}' on NSE or BSE (Yahoo)."], status="error")


# --------------------------------------------------------------------------- pure calculations

def momentum_from_closes(closes):
    """1w/1m/3m price change and a trend label from closes (oldest first)."""
    closes = [float(c) for c in closes]
    n = len(closes)
    out = {}
    if n < 2:
        return out
    cur = closes[-1]
    if n >= 6:
        out["1w_change_pct"] = round((cur - closes[-6]) / closes[-6] * 100, 2)
    if n >= 21:
        out["1m_change_pct"] = round((cur - closes[-21]) / closes[-21] * 100, 2)
    out["3m_change_pct"] = round((cur - closes[0]) / closes[0] * 100, 2)

    m1 = out.get("1m_change_pct", 0) or 0
    m3 = out.get("3m_change_pct", 0) or 0
    if m1 > 5 and m3 > 10:
        out["trend_label"] = "Strong Uptrend"
    elif m1 < -5 and m3 < -10:
        out["trend_label"] = "Downtrend"
    elif abs(m1) < 3 and abs(m3) < 5:
        out["trend_label"] = "Sideways"
    elif m1 > 0 and m3 > 0:
        out["trend_label"] = "Weak Uptrend"
    else:
        out["trend_label"] = "Weak Downtrend"
    return out


def dividend_yield_pct(info, price):
    """Dividend yield in percent. Returns (value_or_None, warning_or_None).

    Yahoo (yfinance >= 1.0) already reports dividendYield in percent (0.34 means 0.34%),
    so it must not be multiplied by 100. When both rate and price exist, rate/price is
    used instead because it is unambiguous.
    """
    rate, price = _num(info.get("dividendRate")), _num(price)
    if rate and price and price > 0:
        y = rate / price * 100
    else:
        y = _num(info.get("dividendYield"))
    if not y:
        return None, None
    if y > 30:
        return None, f"Dividend yield of {y:.1f}% looks implausible and was dropped."
    return round(y, 2), None


def volatility_pct(closes):
    """Annualized volatility (percent) of daily returns; needs at least 30 closes."""
    s = pd.Series([float(c) for c in closes])
    r = s.pct_change().dropna()
    if len(r) < 30:
        return None
    return round(float(r.std() * math.sqrt(TRADING_DAYS) * 100), 2)


def volatility_bucket(vol):
    if vol is None:
        return None
    return "Low" if vol < 20 else "Medium" if vol < 40 else "High"


def moving_average_gap_pct(closes, window=200):
    """Percent the latest close is above (+) or below (-) its `window`-day average; None if too short."""
    c = [float(x) for x in closes if x == x]
    if len(c) < window:
        return None
    ma = sum(c[-window:]) / window
    return round((c[-1] / ma - 1) * 100, 2) if ma else None


def max_drawdown_pct(closes):
    """Largest peak-to-trough fall (negative percent) over the series."""
    s = pd.Series([float(c) for c in closes])
    if len(s) < 2:
        return None
    dd = (s / s.cummax() - 1) * 100
    return round(float(dd.min()), 2)


def beta_vs_index(stock_closes, index_closes, min_obs=60):
    """Beta of a stock against an index from daily returns aligned by date.

    Both arguments are pandas Series indexed by date. Returns (beta_or_None, n_obs).
    """
    r = pd.concat(
        [stock_closes.pct_change(), index_closes.pct_change()], axis=1, join="inner"
    ).dropna()
    r.columns = ["s", "i"]
    if len(r) < min_obs or r["i"].var() == 0:
        return None, len(r)
    return round(float(r.cov().iloc[0, 1] / r["i"].var()), 2), len(r)


def classify_trend(values, tol):
    """'improving' | 'declining' | 'stable' from values ordered newest first."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    delta = vals[0] - vals[-1]
    if abs(delta) <= tol:
        return "stable"
    return "improving" if delta > 0 else "declining"


def _pick(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def _at(series, col):
    if series is None:
        return None
    try:
        return _num(series[col])
    except KeyError:
        return None


FINANCIAL_NOT_APPLICABLE = ["roce_pct", "ebit_margin_pct", "fcf_cr", "de_ratio"]


def compute_fundamentals(income, balance, cash, info=None, is_financial=False, max_years=4):
    """Fundamentals from annual statements (columns = fiscal year ends, newest first).

    Returns (data, warnings). `data` has a 'summary' of latest-year values and trends, and
    'history' with one entry per fiscal year. Money is in INR crore.
    """
    info = info or {}
    warnings = []
    if income is None or getattr(income, "empty", True):
        return {"summary": {}, "history": [], "not_applicable": []}, ["No annual income statement available."]

    rev_s = _pick(income, ["Total Revenue", "Operating Revenue"])
    ebit_s = _pick(income, ["EBIT", "Operating Income"])
    ni_s = _pick(income, ["Net Income", "Net Income Common Stockholders"])
    eps_s = _pick(income, ["Basic EPS", "Diluted EPS"])
    ta_s = _pick(balance, ["Total Assets"])
    cl_s = _pick(balance, ["Current Liabilities"])
    eq_s = _pick(balance, ["Stockholders Equity", "Common Stock Equity"])
    debt_s = _pick(balance, ["Total Debt"])
    ocf_s = _pick(cash, ["Operating Cash Flow"])
    capex_s = _pick(cash, ["Capital Expenditure"])
    fcf_s = _pick(cash, ["Free Cash Flow"])

    history = []
    for col in list(income.columns)[:max_years]:
        rev, ebit, ni = _at(rev_s, col), _at(ebit_s, col), _at(ni_s, col)
        ta, cl, eq, debt = _at(ta_s, col), _at(cl_s, col), _at(eq_s, col), _at(debt_s, col)
        fcf = _at(fcf_s, col)
        if fcf is None:
            ocf, capex = _at(ocf_s, col), _at(capex_s, col)
            fcf = ocf + capex if ocf is not None and capex is not None else None  # capex is negative

        cap_employed = ta - cl if ta is not None and cl is not None else None
        entry = {
            "fy": str(col)[:10],
            "revenue_cr": _r(rev / CRORE, 0) if rev is not None else None,
            "net_income_cr": _r(ni / CRORE, 0) if ni is not None else None,
            "net_margin_pct": _pct(ni, rev),
            "roe_pct": _pct(ni, eq) if eq and eq > 0 else None,
            "eps": _r(_at(eps_s, col)),
            "ebit_margin_pct": _pct(ebit, rev),
            "roce_pct": _pct(ebit, cap_employed) if cap_employed and cap_employed > 0 else None,
            "de_ratio": _r(debt / eq) if debt is not None and eq and eq > 0 else None,
            "fcf_cr": _r(fcf / CRORE, 0) if fcf is not None else None,
        }
        if is_financial:
            for k in FINANCIAL_NOT_APPLICABLE:
                entry[k] = None
        history.append(entry)

    if not history:
        return {"summary": {}, "history": [], "not_applicable": []}, ["Income statement had no fiscal years."]

    latest = history[0]
    revs = [h["revenue_cr"] for h in history]
    eps = [h["eps"] for h in history]

    def yoy(vals):
        if len(vals) >= 2 and vals[0] is not None and vals[1] not in (None, 0) and vals[1] > 0:
            return round((vals[0] / vals[1] - 1) * 100, 2)
        return None

    cagr = None
    valid_revs = [v for v in revs if v is not None]
    if len(valid_revs) >= 3 and valid_revs[-1] > 0 and valid_revs[0] > 0:
        cagr = round(((valid_revs[0] / valid_revs[-1]) ** (1 / (len(valid_revs) - 1)) - 1) * 100, 2)

    fcfs = [h["fcf_cr"] for h in history]
    fcf_tol = abs(fcfs[-1]) * 0.10 if fcfs and fcfs[-1] is not None else 0

    roe_ttm = _r(_num(info.get("returnOnEquity")) * 100) if _num(info.get("returnOnEquity")) is not None else None

    summary = {
        "latest_fy": latest["fy"],
        "years_available": len(history),
        "revenue_cr": latest["revenue_cr"],
        "revenue_growth_pct": yoy(revs),
        "revenue_cagr_pct": cagr,
        "eps_growth_pct": yoy(eps),
        "net_margin_pct": latest["net_margin_pct"],
        "roe_pct": latest["roe_pct"],
        "roe_pct_yahoo": roe_ttm,
        "roce_pct": latest["roce_pct"],
        "ebit_margin_pct": latest["ebit_margin_pct"],
        "de_ratio": latest["de_ratio"],
        "fcf_cr": latest["fcf_cr"],
        "roce_trend": classify_trend([h["roce_pct"] for h in history], 2.0),
        "margin_trend": classify_trend([h["ebit_margin_pct"] for h in history], 1.0),
        "fcf_trend": classify_trend(fcfs, fcf_tol),
    }

    not_applicable = []
    if is_financial:
        not_applicable = list(FINANCIAL_NOT_APPLICABLE)
        warnings.append(
            "Bank/NBFC/insurer: ROCE, EBIT margin, FCF and D/E are not meaningful for financials "
            "and were not computed; use ROE, P/B and growth. GNPA/NIM/CASA are not available from Yahoo."
        )
        for k in FINANCIAL_NOT_APPLICABLE + ["roce_trend", "margin_trend", "fcf_trend"]:
            summary[k] = None
    else:
        if latest["de_ratio"] is not None and latest["de_ratio"] > 5:
            warnings.append(f"D/E of {latest['de_ratio']} is very high; it may include lease liabilities.")
    if latest["roe_pct"] is not None and abs(latest["roe_pct"]) > 100:
        warnings.append(f"ROE of {latest['roe_pct']}% is distorted by very small or negative equity.")
    if (latest["roe_pct"] is not None and roe_ttm is not None and abs(latest["roe_pct"] - roe_ttm) > 5):
        warnings.append(
            f"Statement-based ROE ({latest['roe_pct']}%) differs from Yahoo's ({roe_ttm}%); "
            "the periods or equity definitions differ, so treat both with caution."
        )
    ni_latest, rev_latest, fcf_latest = latest["net_income_cr"], latest["revenue_cr"], latest["fcf_cr"]
    if (not is_financial and fcf_latest is not None and ni_latest is not None and rev_latest
            and ni_latest < 0 and fcf_latest > 0.15 * rev_latest):
        warnings.append(
            f"FCF of {fcf_latest:,.0f} Cr despite a net loss; often caused by lease or "
            "working-capital effects. Verify before relying on it."
        )
    if len(history) < 3:
        warnings.append(f"Only {len(history)} fiscal year(s) available; trends are indicative.")

    return {"summary": summary, "history": history, "not_applicable": not_applicable}, warnings


def earnings_label(days):
    """Label for days until the next results (negative = already reported).

    The original code used abs(days), so results published six days ago read as "This week".
    """
    if 0 <= days <= 7:
        return "This week"
    if 7 < days <= 14:
        return "Within 2 weeks"
    if -30 <= days < 0:
        return "Recently reported"
    return "Not soon"


def _earnings_proximity(info):
    ts = info.get("earningsTimestamp")
    if not ts:
        return "Unknown"
    try:
        return earnings_label((datetime.fromtimestamp(ts) - datetime.now()).days)
    except Exception:
        return "Unknown"


# --------------------------------------------------------------------------- the tools

def get_price_momentum(ticker):
    """Price, 52-week range and short-term momentum for an Indian stock.

    Args:
        ticker: NSE symbol, e.g. 'TCS' or 'BAJAJ-AUTO'.
    Returns an envelope whose data has current_price, day_high, day_low, 52w_high, 52w_low,
    pct_from_52w_high, pct_from_52w_low, drawdown_from_high, 1w/1m/3m_change_pct, trend_label.
    """
    source = "Yahoo Finance via yfinance: quote + 3 months of daily prices"
    loc = _locate(ticker)
    if not loc:
        return _not_found(ticker, source)
    ysym, exchange, info = loc
    warnings, data = [], {}

    price = _r(info.get("currentPrice") or info.get("regularMarketPrice"))
    data["current_price"] = price
    data["day_high"] = _r(info.get("dayHigh"))
    data["day_low"] = _r(info.get("dayLow"))
    data["52w_high"] = _r(info.get("fiftyTwoWeekHigh"))
    data["52w_low"] = _r(info.get("fiftyTwoWeekLow"))

    hi, lo = data["52w_high"], data["52w_low"]
    data["pct_from_52w_high"] = _pct(price - hi, hi) if price and hi else None
    data["drawdown_from_high"] = abs(data["pct_from_52w_high"]) if data["pct_from_52w_high"] is not None else None
    data["pct_from_52w_low"] = _pct(price - lo, lo) if price and lo else None

    hist = _get_history(ysym, "3mo")
    if hist is None:
        warnings.append("Price history unavailable; momentum fields missing.")
    else:
        data.update(momentum_from_closes(hist["Close"].tolist()))
        data["price_as_of"] = str(hist.index[-1])[:10]

    data["exchange"] = exchange
    return _envelope(clean_ticker_symbol(ticker), ysym, source, data, warnings)


def get_valuation(ticker):
    """Valuation ratios and dividend information for an Indian stock.

    Args:
        ticker: NSE symbol, e.g. 'TCS'.
    Returns an envelope whose data has pe_ratio, pb_ratio, peg_ratio, market_cap_cr,
    dividend_yield (percent) and recent_dividend.
    """
    source = "Yahoo Finance via yfinance: quote summary"
    loc = _locate(ticker)
    if not loc:
        return _not_found(ticker, source)
    ysym, exchange, info = loc
    warnings = []

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    dy, dy_warning = dividend_yield_pct(info, price)
    if dy_warning:
        warnings.append(dy_warning)

    mcap = _num(info.get("marketCap"))
    data = {
        "pe_ratio": _r(info.get("trailingPE")),
        "pb_ratio": _r(info.get("priceToBook")),
        "peg_ratio": _r(info.get("pegRatio")),
        "market_cap_cr": _r(mcap / CRORE) if mcap else None,
        "dividend_yield": dy,
        "recent_dividend": bool(info.get("exDividendDate")),
        "exchange": exchange,
    }
    return _envelope(clean_ticker_symbol(ticker), ysym, source, data, warnings)


def get_fundamentals(ticker):
    """Business quality from annual financial statements: ROCE, ROE, margins, growth, FCF, debt.

    Banks, NBFCs and insurers are handled separately: ROCE, EBIT margin, FCF and D/E are
    reported as not applicable instead of being computed.

    Args:
        ticker: NSE symbol, e.g. 'TITAN'.
    Returns an envelope whose data has 'summary' (latest year + trends), 'history' (per fiscal
    year, newest first, money in INR crore) and 'not_applicable'.
    """
    source = "Yahoo Finance via yfinance: annual income statement, balance sheet, cash flow"
    loc = _locate(ticker)
    if not loc:
        return _not_found(ticker, source)
    ysym, exchange, info = loc

    stmts = _get_statements(ysym)
    if not stmts:
        return _envelope(clean_ticker_symbol(ticker), ysym, source,
                         {"summary": {}, "history": [], "not_applicable": []},
                         ["Could not download financial statements."], status="error")

    income, balance, cash = stmts
    is_financial = info.get("sector") == "Financial Services"
    data, warnings = compute_fundamentals(income, balance, cash, info=info, is_financial=is_financial)
    data["is_financial"] = is_financial
    data["industry"] = info.get("industry")
    status = "error" if not data["history"] else None
    return _envelope(clean_ticker_symbol(ticker), ysym, source, data, warnings, status=status)


def get_risk(ticker):
    """Volatility, beta against NIFTY 50, maximum drawdown and earnings timing.

    Beta is computed from 1 year of daily returns against NIFTY 50 (Yahoo's own beta is not
    measured against the Indian market and is not used).

    Args:
        ticker: NSE symbol, e.g. 'INDIGO'.
    Returns an envelope whose data has volatility_pct, volatility_bucket, beta, beta_obs,
    max_drawdown_1y_pct and earnings_proximity.
    """
    source = "Yahoo Finance via yfinance: 1 year of daily prices (stock and ^NSEI)"
    loc = _locate(ticker)
    if not loc:
        return _not_found(ticker, source)
    ysym, exchange, info = loc
    warnings = []

    data = {
        "volatility_pct": None, "volatility_bucket": None, "beta": None, "beta_obs": 0,
        "beta_benchmark": "NIFTY 50 (^NSEI), 1y daily returns", "max_drawdown_1y_pct": None,
        "pct_vs_200dma": None,
        "earnings_proximity": _earnings_proximity(info), "exchange": exchange,
    }

    hist = _get_history(ysym, "1y")
    if hist is None:
        warnings.append("1y price history unavailable; risk fields missing.")
        return _envelope(clean_ticker_symbol(ticker), ysym, source, data, warnings)

    closes = hist["Close"]
    data["volatility_pct"] = volatility_pct(closes.tolist())
    data["volatility_bucket"] = volatility_bucket(data["volatility_pct"])
    data["max_drawdown_1y_pct"] = max_drawdown_pct(closes.tolist())
    data["pct_vs_200dma"] = moving_average_gap_pct(closes.tolist(), 200)

    index = _get_history("^NSEI", "1y")
    if index is None:
        warnings.append("NIFTY history unavailable; beta missing.")
    else:
        beta, n = beta_vs_index(closes, index["Close"])
        data["beta"], data["beta_obs"] = beta, n
        if beta is None:
            warnings.append(f"Not enough overlapping days ({n}) to compute beta.")
    return _envelope(clean_ticker_symbol(ticker), ysym, source, data, warnings)


# --------------------------------------------------------------------------- composite

def fetch_stock_metrics_yf(ticker):
    """Flat dict of all metrics (backward compatible with the Streamlit UI).

    Returns None when the stock cannot be found on NSE/BSE. Values are scalars only; any
    caveats are joined into 'data_warnings'. Units: *_pct percent, debt_to_equity a ratio.
    """
    momentum = get_price_momentum(ticker)
    if momentum["status"] == "error":
        return None
    valuation, risk, fundamentals = get_valuation(ticker), get_risk(ticker), get_fundamentals(ticker)

    metrics = {}
    metrics.update(momentum["data"])
    metrics.update(valuation["data"])
    metrics.update({k: v for k, v in risk["data"].items() if k != "beta_benchmark"})

    s = fundamentals["data"].get("summary", {}) if fundamentals["status"] != "error" else {}
    metrics["roe"] = s.get("roe_pct_yahoo") if s.get("roe_pct_yahoo") is not None else s.get("roe_pct")
    metrics["debt_to_equity"] = s.get("de_ratio")
    for k in ("roce_pct", "ebit_margin_pct", "net_margin_pct", "revenue_growth_pct", "revenue_cagr_pct",
              "eps_growth_pct", "fcf_cr", "roce_trend", "margin_trend", "fcf_trend"):
        metrics[k] = s.get(k)
    metrics["fundamentals_fy"] = s.get("latest_fy")

    warnings = momentum["warnings"] + valuation["warnings"] + risk["warnings"] + fundamentals["warnings"]
    if warnings:
        metrics["data_warnings"] = " | ".join(dict.fromkeys(warnings))
    return metrics
