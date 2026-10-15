"""The rulebook: transparent, editable signals that FLAG things worth a closer look.

Rules do not give advice. Markets have too many factors for that, and a human decides. A rule raises a
flag with the numbers behind it and says where its threshold comes from, so you can disagree with it.
Flags then decide what the research agent investigates, and a few hard ones act as guardrails on what
the model is allowed to recommend (for example, no ADD on a position that is already over-concentrated).

IMPORTANT: there is no authoritative standard for any of these thresholds. They are common screening
conventions and rules of thumb, collected from screener platforms and adviser guidance, and several
are this project's own heuristics. Each rule's `provenance` says which. Change them in THRESHOLDS.
"""
from dataclasses import dataclass

# --------------------------------------------------------------------------- editable thresholds

THRESHOLDS = {
    "roce_min_pct": 15.0,               # screener presets commonly use ROCE >= 15%
    "de_max": 1.0,                      # screener presets commonly use Debt/Equity <= 1
    "below_200dma_pct": 0.0,            # price below its 200-day average is the usual bearish trend rule
    "drawdown_from_high_pct": -20.0,    # 20% below the 52-week high is commonly called a correction
    "lag_sector_1m_pp": -5.0,           # this project's own heuristic
    "beta_high": 1.3,                   # this project's own heuristic
    "volatility_high_pct": 40.0,        # the original app's "High" volatility bucket
    "strong_vs_sector_1m_pp": 3.0,      # this project's own heuristic
    "stock_weight_watch_pct": 10.0,     # adviser guidance: above ~10% in one stock deserves attention
    "stock_weight_high_pct": 20.0,      # ...and above ~20% is a serious risk-management priority
    "sector_weight_watch_pct": 25.0,    # common rule of thumb: no more than ~25% in one sector
    "sector_weight_high_pct": 40.0,     # the original app's system.txt treats 40% as urgent
}

_PROV = {
    "screener": "Common screener-platform preset (e.g. Tickertape / Multibagg ROCE and Debt/Equity screens); "
                "not an official standard.",
    "trend": "Widely used trend convention (price below its 200-day average is read as bearish); sources "
             "stress it is not a trading system on its own.",
    "convention": "Common market convention; no authoritative definition.",
    "own": "This project's own heuristic; not from an external source.",
    "adviser": "Adviser rule of thumb; published limits range from about 5% to 25% depending on the source.",
    "app": "Taken from the original app's own rules (system.txt / volatility buckets).",
}

SEVERITY_WEIGHT = {"high": 3, "watch": 2, "info": 1}


@dataclass
class Flag:
    rule: str
    kind: str          # "concern" or "strength"
    severity: str      # "info" | "watch" | "high"
    title: str
    detail: str        # a plain sentence with the numbers, ready to quote
    provenance: str
    scope: str = "stock"
    symbol: str = ""

    def as_dict(self):
        return self.__dict__.copy()


def _n(x):
    """float(x) or None. Strings such as 'not applicable (bank...)' become None."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _na(x):
    return isinstance(x, str) and x.lower().startswith("not applicable")


# --------------------------------------------------------------------------- stock-level rules

def evaluate_stock(symbol, snap, weight_pct=None, vs_sector_1m=None, thresholds=None, n_holdings=None):
    """Flags for one stock from its snapshot (the dict get_stock_snapshot returns).

    `n_holdings` makes position-size limits fair for small portfolios: limits written for diversified
    portfolios (10% / 20%) would flag every position in a 5-stock portfolio, so the effective limit is
    the larger of the absolute figure and a multiple of an equal split.
    """
    t = {**THRESHOLDS, **(thresholds or {})}
    out = []

    def add(rule, kind, sev, title, detail, prov):
        out.append(Flag(rule, kind, sev, title, detail, _PROV[prov], "stock", symbol))

    financial = _na(snap.get("roce_pct"))
    roce, trend, de = _n(snap.get("roce_pct")), snap.get("roce_trend"), _n(snap.get("debt_to_equity"))
    fcf, dma = _n(snap.get("fcf_cr")), _n(snap.get("pct_vs_200dma"))
    dd, beta, vol = _n(snap.get("pct_from_52w_high")), _n(snap.get("beta")), _n(snap.get("volatility_pct"))

    if not financial:
        if roce is not None and roce < t["roce_min_pct"]:
            add("ROCE_LOW", "concern", "watch", "ROCE below the usual screen",
                f"ROCE is {roce}%, below the commonly used {t['roce_min_pct']:.0f}% screen.", "screener")
        if trend == "declining":
            add("ROCE_DECLINING", "concern", "watch", "ROCE trend is declining",
                f"ROCE has been declining (latest {roce}%)." if roce is not None else "ROCE has been declining.", "own")
        if roce is not None and roce >= t["roce_min_pct"] and trend == "improving":
            add("ROCE_STRONG", "strength", "info", "High and improving ROCE",
                f"ROCE is {roce}% and improving.", "screener")
        if de is not None and de > t["de_max"]:
            add("LEVERAGE_HIGH", "concern", "watch", "Debt-to-equity above the usual screen",
                f"Debt-to-equity is {de}, above the commonly used {t['de_max']:g}. Lease-heavy businesses "
                f"(airlines, retail) often look high on this measure.", "screener")
        if fcf is not None and fcf < 0:
            add("FCF_NEGATIVE", "concern", "watch", "Negative free cash flow",
                f"Free cash flow is {fcf:,.0f} crore (negative).", "own")

    if dma is not None:
        if dma < t["below_200dma_pct"]:
            add("BELOW_200DMA", "concern", "watch", "Price below its 200-day average",
                f"The price is {abs(dma)}% below its 200-day average.", "trend")
        else:
            add("ABOVE_200DMA", "strength", "info", "Price above its 200-day average",
                f"The price is {dma}% above its 200-day average.", "trend")
    if dd is not None and dd <= t["drawdown_from_high_pct"]:
        add("DRAWDOWN", "concern", "watch", "Deep fall from the 52-week high",
            f"The price is {abs(dd)}% below its 52-week high.", "convention")
    if vs_sector_1m is not None:
        if vs_sector_1m <= t["lag_sector_1m_pp"]:
            add("LAGS_SECTOR", "concern", "watch", "Lagging its sector",
                f"Over 1 month it is {abs(vs_sector_1m)} percentage points behind its sector benchmark.", "own")
        elif vs_sector_1m >= t["strong_vs_sector_1m_pp"]:
            add("LEADS_SECTOR", "strength", "info", "Ahead of its sector",
                f"Over 1 month it is {vs_sector_1m} percentage points ahead of its sector benchmark.", "own")
    if beta is not None and beta >= t["beta_high"]:
        add("BETA_HIGH", "concern", "info", "High beta",
            f"Beta against NIFTY 50 is {beta}, so it tends to move more than the market.", "own")
    if vol is not None and vol >= t["volatility_high_pct"]:
        add("VOLATILITY_HIGH", "concern", "watch", "High volatility",
            f"Annualised volatility is {vol}%.", "app")
    if str(snap.get("earnings_proximity", "")) in ("This week", "Within 2 weeks"):
        add("EARNINGS_SOON", "concern", "info", "Results due soon",
            f"Earnings are due {str(snap['earnings_proximity']).lower()}, so expect event risk.", "own")

    w = _n(weight_pct)
    if w is not None:
        high, watch = t["stock_weight_high_pct"], t["stock_weight_watch_pct"]
        basis = "above {:.0f}%".format(high)
        if n_holdings and n_holdings > 0:
            equal = 100.0 / n_holdings
            high_rel, watch_rel = 2.0 * equal, 1.5 * equal
            if max(high_rel, high) > high:
                basis = f"more than twice an equal split ({equal:.0f}% each across {n_holdings} holdings)"
            high, watch = max(high, high_rel), max(watch, watch_rel)
        if w >= high:
            add("STOCK_CONCENTRATION", "concern", "high", "Very large single position",
                f"This stock is {w}% of the portfolio, {basis}.", "adviser")
        elif w >= watch:
            add("STOCK_CONCENTRATION", "concern", "watch", "Large single position",
                f"This stock is {w}% of the portfolio, well above an equal split of the holdings.", "adviser")
    return out


# --------------------------------------------------------------------------- portfolio-level rules

def evaluate_portfolio(summary, thresholds=None):
    """Flags about the portfolio as a whole, from summarize_portfolio()'s output."""
    t = {**THRESHOLDS, **(thresholds or {})}
    out = []
    for sector, pct in (summary.get("sector_allocation") or {}).items():
        if pct >= t["sector_weight_high_pct"]:
            out.append(Flag("SECTOR_CONCENTRATION", "concern", "high", "One sector dominates",
                            f"{sector} is {pct}% of the portfolio, above {t['sector_weight_high_pct']:.0f}%.",
                            _PROV["app"], "portfolio", sector))
        elif pct >= t["sector_weight_watch_pct"]:
            out.append(Flag("SECTOR_CONCENTRATION", "concern", "watch", "Large sector exposure",
                            f"{sector} is {pct}% of the portfolio, above {t['sector_weight_watch_pct']:.0f}%.",
                            _PROV["adviser"], "portfolio", sector))
    return out


# --------------------------------------------------------------------------- using the flags

def concern_score(flags):
    """Higher means more worth investigating. Strengths do not count."""
    return sum(SEVERITY_WEIGHT[f.severity] for f in flags if f.kind == "concern")


def top_concerns(flags_by_symbol, n=2):
    """Symbols to investigate: highest concern score first, ignoring stocks with no concerns."""
    ranked = sorted(((concern_score(f), s) for s, f in flags_by_symbol.items()), reverse=True)
    return [s for score, s in ranked if score > 0][:n]


def apply_guardrails(action, flags):
    """Return (action, reason). The model may not recommend ADD against a hard rule.

    Guardrails only ever make a recommendation more cautious.
    """
    concerns = [f for f in flags if f.kind == "concern"]
    if action == "ADD":
        high = [f for f in concerns if f.severity == "high"]
        if high:
            return "HOLD", f"ADD was changed to HOLD by a rule: {high[0].title.lower()} ({high[0].detail})"
        if len(concerns) >= 3:
            return "HOLD", f"ADD was changed to HOLD by a rule: {len(concerns)} separate concerns are flagged."
    return action, ""
