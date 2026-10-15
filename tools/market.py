"""Market and sector benchmark returns, and market regime classification.

Returns are measured over calendar windows (1w = 7 days, 1m = 30 days, 3m = 91 days): the
change from the last close on or before that date. Counting rows breaks when a series is
missing a day, which Yahoo's ETF data sometimes is.

Sector benchmarks come in three kinds, and every result says which one was used:
    index   a real NSE index with usable history on Yahoo
    etf     an ETF that holds the index's stocks, used as a stand-in (proxy)
    basket  equal-weight average of a few large stocks of the sector
    none    no sector benchmark is available (stated, never silently replaced by NIFTY)
"""
import time

import pandas as pd

from tools._yf import yf, YFINANCE_AVAILABLE

LOOKBACK_DAYS = {"1w": 7, "1m": 30, "3m": 91}
CACHE_TTL_SECONDS = 600
MIN_BASKET_MEMBERS = 3

_cache = {}


def clear_cache():
    _cache.clear()


# Real indices that return daily history on Yahoo. (^CNXFMCG, ^CNXAUTO, ^CNXENERGY, ^CNXMETAL,
# ^CNXINFRA, ^CNXFIN and ^CNXREALTY only return today's quote, so they are not used.)
INDEX_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "NIFTY IT": "^CNXIT",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY PHARMA": "^CNXPHARMA",
}

# Keys are the app's sector names (see tools/sector_map.py).
SECTOR_BENCHMARKS = {
    "IT": {"kind": "index", "symbol": "^CNXIT", "name": "NIFTY IT"},
    "Banking": {"kind": "index", "symbol": "^NSEBANK", "name": "NIFTY BANK"},
    "Pharma": {"kind": "index", "symbol": "^CNXPHARMA", "name": "NIFTY PHARMA"},
    "FMCG": {"kind": "etf", "symbol": "FMCGIETF.NS", "name": "Nifty FMCG ETF (ICICI Prudential)"},
    "Auto": {"kind": "etf", "symbol": "AUTOIETF.NS", "name": "Nifty Auto ETF (ICICI Prudential)"},
    "Energy": {"kind": "etf", "symbol": "ENERGY.NS", "name": "Nifty Energy ETF (Mirae Asset)"},
    "Metal": {"kind": "etf", "symbol": "METALIETF.NS", "name": "Nifty Metal ETF (ICICI Prudential)"},
    "Infrastructure": {"kind": "etf", "symbol": "INFRAIETF.NS", "name": "Nifty Infrastructure ETF (ICICI Prudential)"},
    "Financial Services": {
        "kind": "etf", "symbol": "FINIETF.NS",
        "name": "Nifty Financial Services ex-Bank ETF (ICICI Prudential)",
    },
    "Defence": {"kind": "etf", "symbol": "MODEFENCE.NS", "name": "Nifty India Defence ETF (Motilal Oswal)"},
    "Consumer Cyclical": {
        "kind": "etf", "symbol": "CONSUMBEES.NS",
        "name": "Nifty India Consumption ETF (Nippon), a broad consumption proxy",
    },
    "Chemicals": {
        "kind": "basket", "name": "Chemicals basket",
        "symbols": ["PIDILITIND", "SRF", "UPL", "DEEPAKNTR", "NAVINFLUOR", "TATACHEM", "AARTIIND"],
    },
    "Utilities": {
        "kind": "basket", "name": "Utilities basket",
        "symbols": ["NTPC", "POWERGRID", "TATAPOWER", "NHPC", "ADANIPOWER", "TORNTPOWER"],
    },
    "Real Estate": {
        "kind": "basket", "name": "Real estate basket",
        "symbols": ["DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "LODHA", "PHOENIXLTD"],
    },
    "Communication Services": {
        "kind": "basket", "name": "Telecom basket",
        "symbols": ["BHARTIARTL", "INDUSTOWER", "TATACOMM", "IDEA"],
    },
}

# Sector names that share another sector's benchmark.
SECTOR_ALIASES = {"Oil & Gas": "Energy", "Cement": "Infrastructure"}

NO_BENCHMARK_NOTES = {
    "Aviation": "No sector benchmark: only one large listed airline, so a basket would just be the stock itself.",
}


# --------------------------------------------------------------------------- calculations

def returns_by_calendar(closes, lookbacks=None):
    """% change from the last close on or before (last date - N days), for each window.

    `closes` is a pandas Series indexed by date. A window is omitted when the series does not
    reach back far enough.
    """
    lookbacks = lookbacks or LOOKBACK_DAYS
    closes = closes.dropna().sort_index()
    if len(closes) < 2:
        return {}
    last_date, last = closes.index[-1], float(closes.iloc[-1])
    out = {}
    for key, days in lookbacks.items():
        past = closes[closes.index <= last_date - pd.Timedelta(days=days)]
        if len(past) == 0:
            continue
        base = float(past.iloc[-1])
        if base:
            out[key] = round((last / base - 1) * 100, 2)
    return out


def basket_returns(member_returns, min_members=MIN_BASKET_MEMBERS):
    """Equal-weight average of members' returns, per window.

    `member_returns` is a list of {window: pct} dicts. A window is reported only when at least
    `min_members` members have it. Returns (returns, members_used_per_window).
    """
    out, used = {}, {}
    for key in LOOKBACK_DAYS:
        vals = [m[key] for m in member_returns if key in m and m[key] is not None]
        used[key] = len(vals)
        if len(vals) >= min_members:
            out[key] = round(sum(vals) / len(vals), 2)
    return out, used


# --------------------------------------------------------------------------- data access

def _get_closes(symbol):
    """Daily closes (date-indexed, timezone-free) or None. Cached for CACHE_TTL_SECONDS."""
    hit = _cache.get(symbol)
    now = time.time()
    if hit and now - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]
    if not YFINANCE_AVAILABLE or yf is None:
        return None
    try:
        c = yf.Ticker(symbol).history(period="6mo")["Close"].dropna()
        if len(c) < 2:
            return None
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        _cache[symbol] = (now, c)
        return c
    except Exception:
        return None


def fetch_index_returns(index_name="NIFTY 50"):
    """1w / 1m / 3m returns for NIFTY 50, NIFTY IT, NIFTY BANK or NIFTY PHARMA.

    Returns a dict, or None when the index is unknown or its data is unavailable.
    """
    symbol = INDEX_SYMBOLS.get(index_name)
    if not symbol:
        return None
    closes = _get_closes(symbol)
    if closes is None:
        return None
    return returns_by_calendar(closes) or None


def _config_for(sector):
    sector = SECTOR_ALIASES.get(sector, sector)
    return sector, SECTOR_BENCHMARKS.get(sector)


def get_sector_benchmark(sector):
    """Benchmark returns for a sector, and exactly which benchmark produced them.

    Returns {"sector", "available", "kind", "name", "returns", "note"}; kind is one of
    index | etf | basket | none. When no benchmark exists, available is False and note says why.
    """
    resolved, cfg = _config_for(sector)
    base = {"sector": sector, "available": False, "kind": "none", "name": None, "returns": {}, "note": ""}

    if cfg is None:
        base["note"] = NO_BENCHMARK_NOTES.get(
            sector, f"No sector benchmark is defined for '{sector}'; not compared against the whole market."
        )
        return base

    alias_note = f"'{sector}' uses the {resolved} benchmark. " if resolved != sector else ""

    if cfg["kind"] in ("index", "etf"):
        closes = _get_closes(cfg["symbol"])
        rets = returns_by_calendar(closes) if closes is not None else {}
        if not rets:
            base.update(kind=cfg["kind"], name=cfg["name"],
                        note=f"{alias_note}Benchmark data for {cfg['symbol']} is currently unavailable.")
            return base
        proxy = " This ETF is a stand-in for the index." if cfg["kind"] == "etf" else ""
        base.update(available=True, kind=cfg["kind"], name=cfg["name"], returns=rets,
                    note=f"{alias_note}{cfg['symbol']}.{proxy}".strip())
        return base

    # basket
    members, per_member = [], []
    for sym in cfg["symbols"]:
        closes = _get_closes(sym + ".NS")
        if closes is not None:
            r = returns_by_calendar(closes)
            if r:
                members.append(sym)
                per_member.append(r)
    rets, used = basket_returns(per_member)
    if not rets:
        base.update(kind="basket", name=cfg["name"],
                    note=f"{alias_note}Fewer than {MIN_BASKET_MEMBERS} basket stocks had data.")
        return base
    base.update(available=True, kind="basket", name=f"{cfg['name']} ({len(members)} stocks, equal-weight)",
                returns=rets,
                note=f"{alias_note}Average of: {', '.join(members)}. Not an official index.".strip())
    return base


# --------------------------------------------------------------------------- market regime

# Market Regime Classification
def classify_market_regime(nifty_returns, sector_breadth_data=None):
    """Classify current market regime based on NIFTY performance and breadth"""

    if not nifty_returns:
        return "Unknown", "Insufficient data to classify market regime"

    nifty_1m = nifty_returns.get('1m', 0)
    nifty_3m = nifty_returns.get('3m', 0)

    # Regime classification logic
    if nifty_1m > 3 and nifty_3m > 8:
        regime = "Risk-On"
        description = "Broad market participation with strong upward momentum. Favorable for holding cyclical and growth sectors."
    elif nifty_1m > 1 and nifty_3m > 3:
        regime = "Selective Risk-On"
        description = "Market showing positive momentum but participation is sector-specific. Focus on leading sectors."
    elif abs(nifty_1m) < 2 and abs(nifty_3m) < 5:
        regime = "Range-bound"
        description = "Market lacks clear direction. Favor quality over momentum, reduce position sizing."
    elif nifty_1m < -3 or nifty_3m < -8:
        regime = "Risk-Off"
        description = "Defensive positioning warranted. Prioritize quality, reduce cyclical exposure, raise cash."
    else:
        regime = "Mixed"
        description = "Market showing mixed signals. Be selective, favor sectors with policy tailwinds."

    # ISSUE 2 FIX: Add detailed explanation of how regime is calculated
    explanation = f"\n\n**How we calculate this regime:** Based on NIFTY 50 returns over 1 Month ({nifty_1m:+.2f}%) and 3 Months ({nifty_3m:+.2f}%). "
    explanation += f"**Risk-On** requires 1M > +3% AND 3M > +8%. "
    explanation += f"**Range-bound** means ±2% (1-month) and ±5% (3-month). "
    explanation += f"**Risk-Off** triggers when 1M < -3% OR 3M < -8%."

    return regime, description + explanation
