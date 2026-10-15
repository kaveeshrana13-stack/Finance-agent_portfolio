"""Portfolio helpers: CSV column detection and benchmark-aware portfolio metrics."""
import time
from dataclasses import dataclass, field

import pandas as pd

from tools._yf import YFINANCE_AVAILABLE
from tools.market import fetch_index_returns, get_sector_benchmark
from tools.stocks import fetch_stock_metrics_yf


def empty_portfolio_df():
    return pd.DataFrame(columns=["stock_name", "ticker", "quantity", "avg_buy_price", "sector"])

def smart_detect_csv_columns(cols):
    stock_kw = ['stock', 'name', 'symbol', 'ticker', 'company']
    qty_kw = ['qty', 'quantity', 'shares', 'units', 'nos']
    price_kw = ['price', 'avg', 'average', 'buy', 'cost']
    
    norm = {c.lower().strip().replace('_', '').replace(' ', ''): c for c in cols}
    detected = {'stock': None, 'quantity': None, 'price': None}
    
    for n, o in norm.items():
        if not detected['stock'] and any(k in n for k in stock_kw) and not any(k in n for k in qty_kw):
            detected['stock'] = o
        if not detected['quantity'] and any(k in n for k in qty_kw):
            detected['quantity'] = o
        if not detected['price'] and any(k in n for k in price_kw) and 'current' not in n:
            detected['price'] = o
    
    if not detected['stock'] and cols: detected['stock'] = cols[0]
    if not detected['quantity'] and len(cols) > 1: detected['quantity'] = cols[1]
    if not detected['price'] and len(cols) > 2: detected['price'] = cols[2]
    
    return detected


def _num(x):
    """float(x), or None for None / NaN / non-numeric."""
    v = pd.to_numeric(x, errors="coerce")
    return None if pd.isna(v) else float(v)


@dataclass
class PortfolioMetricsResult:
    df: "pd.DataFrame | None"
    failed_stocks: list = field(default_factory=list)
    successful: int = 0
    stock_metrics: dict = field(default_factory=dict)


def compute_portfolio_metrics(portfolio_df, on_progress=None, pause_seconds=0.5):
    """Fetch fundamentals for every holding and add benchmark, P&L and weight columns.

    on_progress(idx, total, row) is called before each stock is fetched, so a UI can
    show progress. Returns None for an empty portfolio; otherwise a PortfolioMetricsResult
    whose .df is None when no stock could be fetched. pause_seconds spaces out Yahoo requests.
    """
    if portfolio_df.empty:
        return None

    # Fetch NIFTY 50 benchmark once
    nifty_returns = fetch_index_returns("NIFTY 50")

    results = []
    stock_metrics = {}
    successful = 0
    failed_stocks = []
    total = len(portfolio_df)

    # First pass: get all metrics
    for idx, row in portfolio_df.iterrows():
        if on_progress:
            on_progress(idx, total, row)

        metrics = fetch_stock_metrics_yf(row['ticker']) if YFINANCE_AVAILABLE else None

        if metrics:
            successful += 1
            result = {
                "stock_name": row['stock_name'],
                "ticker": row['ticker'],
                "sector": row['sector'],
                "quantity": row['quantity'],
                "avg_buy_price": row['avg_buy_price']
            }
            result.update(metrics)

            # Sector benchmark: a real index, an ETF stand-in, a basket, or none (never NIFTY by default)
            sector_bench = get_sector_benchmark(row['sector'])

            # Add benchmark data
            if nifty_returns:
                result['nifty_1w'] = nifty_returns.get('1w')
                result['nifty_1m'] = nifty_returns.get('1m')
                result['nifty_3m'] = nifty_returns.get('3m')

            if sector_bench["available"]:
                sector_returns = sector_bench["returns"]
                result['sector_1w'] = sector_returns.get('1w')
                result['sector_1m'] = sector_returns.get('1m')
                result['sector_3m'] = sector_returns.get('3m')
                result['sector_index_name'] = sector_bench["name"]
                result['sector_benchmark_kind'] = sector_bench["kind"]
            else:
                result['sector_benchmark_note'] = sector_bench["note"]

            # Calculate relative performance
            stock_1m = result.get('1m_change_pct')

            if stock_1m is not None and result.get('sector_1m') is not None:
                result['vs_sector_1m'] = round(stock_1m - result['sector_1m'], 2)

            if stock_1m is not None and result.get('nifty_1m') is not None:
                result['vs_nifty_1m'] = round(stock_1m - result['nifty_1m'], 2)

            # Relative performance verdict
            if result.get('vs_sector_1m') is not None:
                if result['vs_sector_1m'] > 2:
                    result['sector_verdict'] = "Outperforming"
                elif result['vs_sector_1m'] < -2:
                    result['sector_verdict'] = "Underperforming"
                else:
                    result['sector_verdict'] = "In-line"

            if result.get('vs_nifty_1m') is not None:
                if result['vs_nifty_1m'] > 2:
                    result['market_verdict'] = "Outperforming"
                elif result['vs_nifty_1m'] < -2:
                    result['market_verdict'] = "Underperforming"
                else:
                    result['market_verdict'] = "In-line"

            # Current value needs only a price and a quantity; P&L additionally needs a buy price
            price, qty, cost = _num(metrics.get('current_price')), _num(row['quantity']), _num(row['avg_buy_price'])
            if price is not None and qty is not None and qty > 0:
                result['current_value'] = price * qty
                if cost is not None and cost > 0:
                    result['total_investment'] = cost * qty
                    result['unrealized_pl'] = result['current_value'] - result['total_investment']
                    result['unrealized_pl_pct'] = (result['unrealized_pl'] / result['total_investment']) * 100

            results.append(result)
            stock_metrics[row['ticker']] = metrics
        else:
            failed_stocks.append(row['stock_name'])

        time.sleep(pause_seconds)

    if not results:
        return PortfolioMetricsResult(None, failed_stocks, successful, stock_metrics)

    result_df = pd.DataFrame(results)

    # Calculate portfolio-level context for each stock
    total_portfolio_value = result_df['current_value'].sum() if 'current_value' in result_df.columns else 0

    for idx in result_df.index:
        stock_value = result_df.loc[idx, 'current_value'] if 'current_value' in result_df.columns else 0
        sector = result_df.loc[idx, 'sector']

        # Stock weight
        if total_portfolio_value > 0:
            result_df.loc[idx, 'portfolio_weight_pct'] = (stock_value / total_portfolio_value) * 100

        # Sector weight
        sector_total = result_df[result_df['sector'] == sector]['current_value'].sum() if 'current_value' in result_df.columns else 0
        if total_portfolio_value > 0:
            result_df.loc[idx, 'sector_weight_pct'] = (sector_total / total_portfolio_value) * 100

    return PortfolioMetricsResult(result_df, failed_stocks, successful, stock_metrics)


def summarize_portfolio(df, top_n=5):
    """Portfolio-level numbers from the table compute_portfolio_metrics() returns.

    Pure and deterministic: nothing here calls an LLM. Holdings that cannot be valued (no
    quantity or no price) are excluded from totals and listed in `data_quality` instead of
    silently skewing them. The overall return is like-for-like: only holdings with both a
    current value and a known cost are compared.
    """
    empty = {
        "holdings": 0, "total_value": 0.0, "value_of_priced_holdings": 0.0, "cost_of_priced_holdings": 0.0,
        "overall_return_pct": None, "sector_allocation": {}, "top_holdings": [],
        "concentration": {"top_holding_name": None, "top_holding_pct": 0.0, "top5_pct": 0.0,
                          "largest_sector": None, "largest_sector_pct": 0.0},
        "beta_value_weighted": None, "beta_coverage_pct": 0.0,
        "momentum": {"uptrend": 0, "downtrend": 0, "sideways": 0, "unknown": 0, "positive_1m_pct": None},
        "volatility": {"high_count": 0, "total": 0}, "data_quality": ["Portfolio is empty."],
    }
    if df is None or len(df) == 0:
        return empty

    def col(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(float("nan"), index=df.index)

    value, cost, qty, beta = col("current_value"), col("total_investment"), col("quantity"), col("beta")
    names = df["stock_name"] if "stock_name" in df.columns else df["ticker"]
    valued = value.notna()
    notes = []

    # holdings that cannot be valued
    no_qty = df.index[~valued & ~(qty > 0)]
    no_price = df.index[~valued & (qty > 0)]
    if len(no_qty):
        notes.append(f"{len(no_qty)} holding(s) have no quantity and are excluded from totals and weights: "
                     + ", ".join(str(names[i]) for i in no_qty))
    if len(no_price):
        notes.append(f"{len(no_price)} holding(s) have no live price and are excluded from totals and weights: "
                     + ", ".join(str(names[i]) for i in no_price))

    total_value = float(value[valued].sum())

    # like-for-like return: only holdings with both a value and a known cost
    both = valued & cost.notna()
    value_known, cost_known = float(value[both].sum()), float(cost[both].sum())
    overall = round((value_known - cost_known) / cost_known * 100, 2) if cost_known > 0 else None
    no_cost = df.index[valued & cost.isna()]
    if len(no_cost):
        notes.append(f"{len(no_cost)} holding(s) have no buy price; the overall return covers only the other "
                     f"{int(both.sum())} holding(s): " + ", ".join(str(names[i]) for i in no_cost))

    # allocation and concentration (valued holdings only)
    sectors = df["sector"].fillna("Unknown") if "sector" in df.columns else pd.Series("Unknown", index=df.index)
    allocation, top, top_n_pct = {}, [], 0.0
    if total_value > 0:
        by_sector = value[valued].groupby(sectors[valued]).sum().sort_values(ascending=False)
        allocation = {str(k): round(float(v) / total_value * 100, 2) for k, v in by_sector.items()}
        order = value[valued].sort_values(ascending=False).head(top_n)
        top_n_pct = round(float(order.sum()) / total_value * 100, 2)      # from unrounded values
        for i, v in order.items():
            top.append({"name": str(names[i]), "ticker": str(df.at[i, "ticker"]) if "ticker" in df.columns else None,
                        "value": round(float(v), 2), "weight_pct": round(float(v) / total_value * 100, 2)})

    largest_sector = next(iter(allocation), None)
    concentration = {
        "top_holding_name": top[0]["name"] if top else None,
        "top_holding_pct": top[0]["weight_pct"] if top else 0.0,
        "top5_pct": top_n_pct,
        "largest_sector": largest_sector,
        "largest_sector_pct": allocation.get(largest_sector, 0.0) if largest_sector else 0.0,
    }

    # value-weighted beta (a 40% position counts four times a 10% one)
    with_beta = valued & beta.notna()
    beta_vw, beta_cov = None, 0.0
    if with_beta.any() and total_value > 0:
        w = value[with_beta]
        beta_vw = round(float((beta[with_beta] * w).sum() / w.sum()), 2)
        beta_cov = round(float(w.sum()) / total_value * 100, 1)
        if beta_cov < 99.9:
            notes.append(f"Beta covers only {beta_cov:.0f}% of portfolio value.")

    # momentum breadth and volatility
    trend = df["trend_label"].fillna("") if "trend_label" in df.columns else pd.Series("", index=df.index)
    up, down = trend.str.contains("Uptrend"), trend.str.contains("Downtrend")
    side = trend.str.contains("Sideways")
    m1 = col("1m_change_pct")
    positive_1m = round(float((m1[m1.notna()] > 0).mean() * 100), 1) if m1.notna().any() else None
    vol = df["volatility_bucket"] if "volatility_bucket" in df.columns else pd.Series("", index=df.index)

    # duplicate tickers are separate lots, not merged
    if "ticker" in df.columns:
        dup = df["ticker"][df["ticker"].duplicated(keep=False) & df["ticker"].astype(str).str.len().gt(0)]
        if len(dup):
            notes.append("Same stock in several rows (separate lots, not merged): " + ", ".join(sorted(set(map(str, dup)))))

    return {
        "holdings": int(len(df)),
        "total_value": round(total_value, 2),
        "value_of_priced_holdings": round(value_known, 2),
        "cost_of_priced_holdings": round(cost_known, 2),
        "overall_return_pct": overall,
        "sector_allocation": allocation,
        "top_holdings": top,
        "concentration": concentration,
        "beta_value_weighted": beta_vw,
        "beta_coverage_pct": beta_cov,
        "momentum": {"uptrend": int(up.sum()), "downtrend": int(down.sum()), "sideways": int(side.sum()),
                     "unknown": int((~(up | down | side)).sum()), "positive_1m_pct": positive_1m},
        "volatility": {"high_count": int((vol == "High").sum()), "total": int(len(df))},
        "data_quality": notes,
    }
