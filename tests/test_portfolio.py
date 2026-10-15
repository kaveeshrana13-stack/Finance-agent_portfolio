"""Offline tests for portfolio enrichment and the portfolio summary (market data is stubbed)."""
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.portfolio as P
from tools.portfolio import compute_portfolio_metrics, summarize_portfolio

NO_BENCH = {"available": False, "kind": "none", "name": None, "returns": {}, "note": "no benchmark"}
FLAT_BENCH = {"available": True, "kind": "index", "name": "TEST", "returns": {"1w": 0.0, "1m": 0.0, "3m": 0.0}, "note": ""}


def table(rows):
    """Build the holdings table the way the app does: an empty frame plus one concat per row."""
    df = P.empty_portfolio_df()
    for name, ticker, qty, price, sector in rows:
        row = {"stock_name": name, "ticker": ticker, "quantity": qty, "avg_buy_price": price, "sector": sector}
        with warnings.catch_warnings():            # pandas deprecation raised by the app's own pattern
            warnings.simplefilter("ignore", FutureWarning)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def enrich(rows, metrics, bench=NO_BENCH, nifty=None):
    with mock.patch.object(P, "fetch_stock_metrics_yf", lambda t: metrics.get(t)), \
         mock.patch.object(P, "fetch_index_returns", lambda n: nifty), \
         mock.patch.object(P, "get_sector_benchmark", lambda s: bench), \
         mock.patch.object(P.time, "sleep", lambda s: None):
        return compute_portfolio_metrics(table(rows))


class Enrichment(unittest.TestCase):
    METRICS = {"AAA": {"current_price": 100.0, "1m_change_pct": 2.0, "beta": 0.5},
               "BBB": {"current_price": 200.0, "1m_change_pct": 0.0, "beta": 1.5},
               "CCC": {"current_price": 50.0, "1m_change_pct": 1.0, "beta": 1.0}}

    def test_value_and_pl(self):
        d = enrich([("A", "AAA", 10, 90.0, "IT")], self.METRICS).df.iloc[0]
        self.assertEqual(d["current_value"], 1000.0)
        self.assertEqual(d["total_investment"], 900.0)
        self.assertAlmostEqual(d["unrealized_pl_pct"], 11.1111, places=3)

    def test_value_exists_without_a_buy_price(self):
        d = enrich([("B", "BBB", 5, None, "IT")], self.METRICS).df.iloc[0]
        self.assertEqual(d["current_value"], 1000.0)
        self.assertTrue(pd.isna(d.get("total_investment")))
        self.assertTrue(pd.isna(d.get("unrealized_pl_pct")))

    def test_missing_quantity_gives_no_value_and_does_not_crash(self):
        r = enrich([("A", "AAA", 10, 90.0, "IT"), ("B", "BBB", None, 180.0, "IT")], self.METRICS)
        b = r.df[r.df.ticker == "BBB"].iloc[0]
        self.assertTrue(pd.isna(b["current_value"]))

    def test_zero_return_is_a_value_not_missing(self):
        """Regression: `if stock_1m and ...` treated 0.0 as missing, so vs_sector was never set."""
        r = enrich([("B", "BBB", 5, 180.0, "IT")], self.METRICS, bench=FLAT_BENCH, nifty={"1w": 0.0, "1m": 0.0, "3m": 0.0})
        row = r.df.iloc[0]
        self.assertEqual(row["vs_sector_1m"], 0.0)
        self.assertEqual(row["vs_nifty_1m"], 0.0)
        self.assertEqual(row["sector_verdict"], "In-line")

    def test_weights_sum_to_100(self):
        r = enrich([("A", "AAA", 10, 90.0, "IT"), ("B", "BBB", 5, 180.0, "IT"), ("C", "CCC", 20, 40.0, "Auto")], self.METRICS)
        self.assertAlmostEqual(r.df["portfolio_weight_pct"].sum(), 100.0, places=6)

    def test_empty_and_unfetchable(self):
        self.assertIsNone(compute_portfolio_metrics(P.empty_portfolio_df()))
        r = enrich([("Z", "ZZZ", 1, 10.0, "IT")], {})
        self.assertIsNone(r.df)
        self.assertEqual(r.failed_stocks, ["Z"])


class Summary(unittest.TestCase):
    def frame(self, **overrides):
        base = {
            "stock_name": ["A", "B", "C"], "ticker": ["AAA", "BBB", "CCC"], "sector": ["IT", "IT", "Auto"],
            "quantity": [10, 5, 20], "current_value": [1000.0, 1000.0, 1000.0],
            "total_investment": [900.0, 900.0, 800.0], "beta": [0.5, 1.5, 1.0],
            "trend_label": ["Weak Uptrend", "Strong Uptrend", "Downtrend"],
            "volatility_bucket": ["High", "Low", "Medium"], "1m_change_pct": [2.0, -1.0, 3.0],
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_clean_portfolio(self):
        s = summarize_portfolio(self.frame())
        self.assertEqual(s["total_value"], 3000.0)
        self.assertEqual(s["overall_return_pct"], round((3000 - 2600) / 2600 * 100, 2))
        self.assertEqual(s["sector_allocation"], {"IT": 66.67, "Auto": 33.33})
        self.assertEqual(s["concentration"]["largest_sector"], "IT")
        self.assertEqual(s["concentration"]["top_holding_pct"], 33.33)
        self.assertEqual(s["concentration"]["top5_pct"], 100.0)
        self.assertEqual(s["momentum"], {"uptrend": 2, "downtrend": 1, "sideways": 0, "unknown": 0, "positive_1m_pct": 66.7})
        self.assertEqual(s["volatility"], {"high_count": 1, "total": 3})
        self.assertEqual(s["data_quality"], [])

    def test_return_is_like_for_like_when_a_buy_price_is_missing(self):
        """Regression: value counted all holdings but cost only some, giving +76.5% instead of +17.6%."""
        s = summarize_portfolio(self.frame(total_investment=[900.0, float("nan"), 800.0]))
        self.assertEqual(s["total_value"], 3000.0)
        self.assertEqual(s["overall_return_pct"], 17.65)           # (2000 - 1700) / 1700
        self.assertNotAlmostEqual(s["overall_return_pct"], 76.5, delta=1)
        self.assertTrue(any("no buy price" in n and "B" in n for n in s["data_quality"]))

    def test_holding_without_quantity_is_listed_not_silently_dropped(self):
        s = summarize_portfolio(self.frame(quantity=[10, float("nan"), 20], current_value=[1000.0, float("nan"), 1000.0],
                                           total_investment=[900.0, float("nan"), 800.0]))
        self.assertEqual(s["total_value"], 2000.0)
        self.assertEqual(s["holdings"], 3)
        self.assertTrue(any("no quantity" in n and "B" in n for n in s["data_quality"]))
        self.assertAlmostEqual(sum(s["sector_allocation"].values()), 100.0, delta=0.05)

    def test_beta_is_value_weighted(self):
        s = summarize_portfolio(self.frame(current_value=[7000.0, 2000.0, 1000.0]))
        self.assertEqual(s["beta_value_weighted"], round((0.5 * 7000 + 1.5 * 2000 + 1.0 * 1000) / 10000, 2))   # 0.75
        self.assertNotEqual(s["beta_value_weighted"], 1.0)         # the simple mean would say 1.0

    def test_partial_beta_coverage_is_reported(self):
        s = summarize_portfolio(self.frame(beta=[0.5, float("nan"), 1.0]))
        self.assertEqual(s["beta_coverage_pct"], 66.7)
        self.assertTrue(any("Beta covers only" in n for n in s["data_quality"]))

    def test_concentration_with_a_dominant_holding(self):
        s = summarize_portfolio(self.frame(current_value=[8000.0, 1000.0, 1000.0]))
        self.assertEqual(s["concentration"]["top_holding_name"], "A")
        self.assertEqual(s["concentration"]["top_holding_pct"], 80.0)
        self.assertEqual(s["top_holdings"][0]["weight_pct"], 80.0)

    def test_top5_is_not_a_sum_of_rounded_weights(self):
        """Regression: three 33.33% weights summed to 99.99 instead of 100.0."""
        s = summarize_portfolio(self.frame())
        self.assertEqual(s["concentration"]["top5_pct"], 100.0)

    def test_duplicate_lots_are_noted_not_merged(self):
        s = summarize_portfolio(self.frame(ticker=["AAA", "AAA", "CCC"]))
        self.assertEqual(s["holdings"], 3)
        self.assertTrue(any("AAA" in n and "separate lots" in n for n in s["data_quality"]))

    def test_empty_and_missing_columns(self):
        self.assertEqual(summarize_portfolio(pd.DataFrame())["data_quality"], ["Portfolio is empty."])
        minimal = pd.DataFrame({"stock_name": ["A"], "ticker": ["AAA"], "current_value": [500.0]})
        s = summarize_portfolio(minimal)                           # no beta/sector/trend columns at all
        self.assertEqual(s["total_value"], 500.0)
        self.assertEqual(s["sector_allocation"], {"Unknown": 100.0})
        self.assertIsNone(s["beta_value_weighted"])
        self.assertIsNone(s["overall_return_pct"])


if __name__ == "__main__":
    unittest.main()
