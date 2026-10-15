"""Portfolio helpers: CSV column detection and benchmark-aware portfolio metrics."""
import time
from dataclasses import dataclass, field

import pandas as pd

from tools._yf import YFINANCE_AVAILABLE
from tools.market import SECTOR_TO_INDEX, fetch_index_returns
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


@dataclass
class PortfolioMetricsResult:
    df: "pd.DataFrame | None"
    failed_stocks: list = field(default_factory=list)
    successful: int = 0
    stock_metrics: dict = field(default_factory=dict)


def compute_portfolio_metrics(portfolio_df, on_progress=None):
    """Fetch fundamentals for every holding and add benchmark, P&L and weight columns.

    on_progress(idx, total, row) is called before each stock is fetched, so a UI can
    show progress. Returns None for an empty portfolio; otherwise a PortfolioMetricsResult
    whose .df is None when no stock could be fetched.
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

            # Fetch sector benchmark
            sector_index = SECTOR_TO_INDEX.get(row['sector'], "NIFTY 50")
            sector_returns = fetch_index_returns(sector_index)

            # Add benchmark data
            if nifty_returns:
                result['nifty_1w'] = nifty_returns.get('1w')
                result['nifty_1m'] = nifty_returns.get('1m')
                result['nifty_3m'] = nifty_returns.get('3m')

            if sector_returns:
                result['sector_1w'] = sector_returns.get('1w')
                result['sector_1m'] = sector_returns.get('1m')
                result['sector_3m'] = sector_returns.get('3m')
                result['sector_index_name'] = sector_index

            # Calculate relative performance
            stock_1m = result.get('1m_change_pct')

            if stock_1m and result.get('sector_1m'):
                result['vs_sector_1m'] = round(stock_1m - result['sector_1m'], 2)

            if stock_1m and result.get('nifty_1m'):
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

            # Calculate unrealized P&L
            if metrics['current_price'] and row['avg_buy_price']:
                result['total_investment'] = row['avg_buy_price'] * row['quantity']
                result['current_value'] = metrics['current_price'] * row['quantity']
                result['unrealized_pl'] = result['current_value'] - result['total_investment']
                result['unrealized_pl_pct'] = (result['unrealized_pl'] / result['total_investment']) * 100

            results.append(result)
            stock_metrics[row['ticker']] = metrics
        else:
            failed_stocks.append(row['stock_name'])

        time.sleep(0.5)

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
