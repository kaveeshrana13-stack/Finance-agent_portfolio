"""Offline tests for the stock metric calculations (no network)."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.stocks import (
    FINANCIAL_NOT_APPLICABLE,
    beta_vs_index,
    classify_trend,
    compute_fundamentals,
    dividend_yield_pct,
    max_drawdown_pct,
    momentum_from_closes,
    volatility_bucket,
    volatility_pct,
)

CR = 1e7  # one crore


def frame(rows, years=("2026-03-31", "2025-03-31", "2024-03-31", "2023-03-31")):
    """Statement-shaped DataFrame: rows are line items, columns are FY ends (newest first)."""
    cols = [pd.Timestamp(y) for y in years[: len(next(iter(rows.values())))]]
    return pd.DataFrame(rows, index=cols).T


class DividendYield(unittest.TestCase):
    """Regression: yields under 1% were multiplied by 100 (TITAN showed 34%)."""

    def test_observed_yahoo_values_are_not_inflated(self):
        cases = [
            ({"dividendYield": 0.34, "dividendRate": None}, 4439.3, 0.34),    # TITAN
            ({"dividendYield": 0.51, "dividendRate": None}, 1204.6, 0.51),    # RELIANCE
            ({"dividendYield": 0.31, "dividendRate": None}, 322.0, 0.31),     # KPIGREEN
            ({"dividendYield": 2.08, "dividendRate": None}, 2357.9, 2.08),    # TCS
            ({"dividendYield": 6.12, "dividendRate": None}, 262.9, 6.12),     # ITC
        ]
        for info, price, expected in cases:
            got, warning = dividend_yield_pct(info, price)
            self.assertEqual(got, expected, info)
            self.assertIsNone(warning)

    def test_rate_over_price_wins_when_available(self):
        # INDIGO: Yahoo yield says 0.0 but the rate is Rs 10 on a Rs 4,788 share
        got, _ = dividend_yield_pct({"dividendYield": 0.0, "dividendRate": 10.0}, 4788.0)
        self.assertEqual(got, 0.21)

    def test_implausible_yield_is_dropped_with_warning(self):
        got, warning = dividend_yield_pct({"dividendYield": 51.0}, 100.0)
        self.assertIsNone(got)
        self.assertIn("implausible", warning)

    def test_no_dividend(self):
        self.assertEqual(dividend_yield_pct({}, 100.0), (None, None))


class Momentum(unittest.TestCase):
    def test_changes_and_trend(self):
        closes = [100.0 + i for i in range(60)]          # steady climb, 60 points
        out = momentum_from_closes(closes)
        self.assertEqual(out["1w_change_pct"], round((159 - 154) / 154 * 100, 2))
        self.assertEqual(out["1m_change_pct"], round((159 - 139) / 139 * 100, 2))
        self.assertEqual(out["3m_change_pct"], 59.0)
        self.assertEqual(out["trend_label"], "Strong Uptrend")

    def test_sideways(self):
        out = momentum_from_closes([100, 101, 100, 99, 100, 101] * 12)
        self.assertEqual(out["trend_label"], "Sideways")

    def test_too_short(self):
        self.assertEqual(momentum_from_closes([100.0]), {})


class Risk(unittest.TestCase):
    def test_beta_of_double_the_index_is_two(self):
        rets = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01] * 20
        dates = pd.date_range("2025-01-02", periods=len(rets))
        index_close = pd.Series((1 + pd.Series(rets, index=dates)).cumprod() * 100)
        stock_rets = pd.Series([2 * r for r in rets], index=dates)
        stock_close = (1 + stock_rets).cumprod() * 100
        beta, n = beta_vs_index(stock_close, index_close)
        self.assertAlmostEqual(beta, 2.0, delta=0.1)
        self.assertGreater(n, 100)

    def test_beta_needs_enough_data(self):
        dates = pd.date_range("2025-01-01", periods=10)
        s = pd.Series(range(100, 110), index=dates, dtype=float)
        beta, n = beta_vs_index(s, s)
        self.assertIsNone(beta)
        self.assertLess(n, 60)

    def test_volatility_and_bucket(self):
        self.assertIsNone(volatility_pct([100.0] * 10))        # too short
        flat = volatility_pct([100.0, 100.0] * 40)
        self.assertEqual(flat, 0.0)
        self.assertEqual(volatility_bucket(0.0), "Low")
        self.assertEqual(volatility_bucket(25.0), "Medium")
        self.assertEqual(volatility_bucket(55.0), "High")
        self.assertIsNone(volatility_bucket(None))

    def test_max_drawdown(self):
        self.assertEqual(max_drawdown_pct([100, 120, 90, 110]), -25.0)   # 120 -> 90
        self.assertEqual(max_drawdown_pct([100, 110, 120]), 0.0)


class Trends(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify_trend([38.4, 36.9, 34.9], 2.0), "improving")   # TITAN ROCE
        self.assertEqual(classify_trend([11.1, 12.9, 19.3], 2.0), "declining")   # KPIGREEN ROCE
        self.assertEqual(classify_trend([9.0, 8.6, 9.3], 2.0), "stable")         # RELIANCE ROCE
        self.assertIsNone(classify_trend([5.0, None], 1.0))


class Fundamentals(unittest.TestCase):
    def setUp(self):
        # Hand-built company; all money in rupees (x CR to get crore).
        self.income = frame({
            "Total Revenue": [1000 * CR, 900 * CR, 800 * CR, 700 * CR],
            "EBIT": [200 * CR, 180 * CR, 150 * CR, 120 * CR],
            "Net Income": [150 * CR, 130 * CR, 100 * CR, 80 * CR],
            "Basic EPS": [15.0, 13.0, 10.0, 8.0],
        })
        self.balance = frame({
            "Total Assets": [1000 * CR, 950 * CR, 900 * CR, 850 * CR],
            "Current Liabilities": [200 * CR, 190 * CR, 180 * CR, 170 * CR],
            "Stockholders Equity": [500 * CR, 450 * CR, 400 * CR, 350 * CR],
            "Total Debt": [250 * CR, 250 * CR, 250 * CR, 250 * CR],
        })
        self.cash = frame({
            "Operating Cash Flow": [180 * CR, 150 * CR, 120 * CR, 100 * CR],
            "Capital Expenditure": [-60 * CR, -50 * CR, -40 * CR, -30 * CR],
        })

    def test_latest_year_values(self):
        data, warnings = compute_fundamentals(self.income, self.balance, self.cash)
        s = data["summary"]
        self.assertEqual(s["roce_pct"], 25.0)            # 200 / (1000 - 200)
        self.assertEqual(s["ebit_margin_pct"], 20.0)     # 200 / 1000
        self.assertEqual(s["net_margin_pct"], 15.0)
        self.assertEqual(s["roe_pct"], 30.0)             # 150 / 500
        self.assertEqual(s["de_ratio"], 0.5)             # 250 / 500, a plain ratio
        self.assertEqual(s["fcf_cr"], 120.0)             # 180 - 60
        self.assertEqual(s["revenue_cr"], 1000.0)
        self.assertEqual(s["latest_fy"], "2026-03-31")
        self.assertEqual(warnings, [])

    def test_growth_and_trends(self):
        s = compute_fundamentals(self.income, self.balance, self.cash)[0]["summary"]
        self.assertEqual(s["revenue_growth_pct"], round((1000 / 900 - 1) * 100, 2))
        self.assertEqual(s["revenue_cagr_pct"], round(((1000 / 700) ** (1 / 3) - 1) * 100, 2))
        self.assertEqual(s["eps_growth_pct"], round((15 / 13 - 1) * 100, 2))
        self.assertEqual(s["margin_trend"], "improving")   # 17.1 -> 20.0
        self.assertEqual(s["fcf_trend"], "improving")
        self.assertEqual(s["years_available"], 4)

    def test_history_is_newest_first(self):
        history = compute_fundamentals(self.income, self.balance, self.cash)[0]["history"]
        self.assertEqual([h["fy"] for h in history][:2], ["2026-03-31", "2025-03-31"])

    def test_financials_are_marked_not_applicable(self):
        income = frame({"Total Revenue": [900 * CR, 800 * CR], "Net Income": [150 * CR, 130 * CR]})
        balance = frame({"Total Assets": [9000 * CR, 8000 * CR], "Stockholders Equity": [1000 * CR, 900 * CR]})
        cash = frame({"Operating Cash Flow": [999 * CR, 888 * CR]})
        data, warnings = compute_fundamentals(income, balance, cash, is_financial=True)
        for key in FINANCIAL_NOT_APPLICABLE:
            self.assertIsNone(data["summary"][key], key)
        self.assertEqual(data["not_applicable"], FINANCIAL_NOT_APPLICABLE)
        self.assertEqual(data["summary"]["roe_pct"], 15.0)           # still reported
        self.assertTrue(any("Bank/NBFC" in w for w in warnings))

    def test_distorted_roe_is_flagged(self):
        balance = self.balance.copy()
        balance.loc["Stockholders Equity"] = [30 * CR, 28 * CR, 25 * CR, 20 * CR]   # tiny equity
        data, warnings = compute_fundamentals(self.income, balance, self.cash)
        self.assertGreater(data["summary"]["roe_pct"], 100)
        self.assertTrue(any("distorted" in w for w in warnings))

    def test_negative_equity_gives_no_ratio(self):
        balance = self.balance.copy()
        balance.loc["Stockholders Equity"] = [-50 * CR, 10 * CR, 20 * CR, 30 * CR]
        s = compute_fundamentals(self.income, balance, self.cash)[0]["summary"]
        self.assertIsNone(s["roe_pct"])
        self.assertIsNone(s["de_ratio"])

    def test_roe_source_disagreement_is_flagged(self):
        info = {"returnOnEquity": 0.45}          # Yahoo says 45%, statements give 30%
        _, warnings = compute_fundamentals(self.income, self.balance, self.cash, info=info)
        self.assertTrue(any("differs from Yahoo" in w for w in warnings))
        close = {"returnOnEquity": 0.31}          # within 5 points: no warning
        _, warnings = compute_fundamentals(self.income, self.balance, self.cash, info=close)
        self.assertFalse(any("differs from Yahoo" in w for w in warnings))

    def test_positive_fcf_despite_loss_is_flagged(self):
        income = self.income.copy()
        income.loc["Net Income"] = [-40 * CR, 130 * CR, 100 * CR, 80 * CR]
        cash = frame({"Free Cash Flow": [300 * CR, 66 * CR, 55 * CR, 44 * CR]})   # 30% of revenue
        _, warnings = compute_fundamentals(income, self.balance, cash)
        self.assertTrue(any("despite a net loss" in w for w in warnings))

    def test_leverage_warning(self):
        balance = self.balance.copy()
        balance.loc["Total Debt"] = [3000 * CR, 2800 * CR, 2500 * CR, 2000 * CR]   # D/E = 6
        _, warnings = compute_fundamentals(self.income, balance, self.cash)
        self.assertTrue(any("lease" in w for w in warnings))

    def test_missing_statements(self):
        data, warnings = compute_fundamentals(None, None, None)
        self.assertEqual(data["history"], [])
        self.assertTrue(warnings)

    def test_explicit_fcf_row_is_used_when_present(self):
        cash = frame({"Free Cash Flow": [77 * CR, 66 * CR, 55 * CR, 44 * CR]})
        s = compute_fundamentals(self.income, self.balance, cash)[0]["summary"]
        self.assertEqual(s["fcf_cr"], 77.0)


if __name__ == "__main__":
    unittest.main()
