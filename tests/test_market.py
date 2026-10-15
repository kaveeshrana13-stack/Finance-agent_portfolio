"""Offline tests for benchmark returns, sector benchmark selection and the regime rule."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.market import (
    INDEX_SYMBOLS,
    MIN_BASKET_MEMBERS,
    SECTOR_ALIASES,
    SECTOR_BENCHMARKS,
    basket_returns,
    classify_market_regime,
    get_sector_benchmark,
    returns_by_calendar,
)
from tools.resolver import _load


def series(values, start="2026-06-01"):
    """Business-day close series ending on the last value."""
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


class CalendarReturns(unittest.TestCase):
    def test_windows_use_calendar_days(self):
        # business days 2026-06-01 .. ; closes rise by 1 per day from 100
        s = series(list(range(100, 160)))
        out = returns_by_calendar(s)
        last_date, last = s.index[-1], s.iloc[-1]
        for key, days in (("1w", 7), ("1m", 30), ("3m", 91)):
            base = s[s.index <= last_date - pd.Timedelta(days=days)]
            if len(base):
                self.assertEqual(out[key], round((last / base.iloc[-1] - 1) * 100, 2), key)

    def test_missing_day_does_not_shift_the_window(self):
        """Regression: an ETF missing one trading day gave different 1w values by row count."""
        full = series(list(range(100, 160)))
        gappy = full.drop(full.index[-3])            # one missing day inside the last week
        a, b = returns_by_calendar(full), returns_by_calendar(gappy)
        self.assertEqual(a["1w"], b["1w"])           # same calendar window, same base price
        self.assertEqual(a["1m"], b["1m"])

    def test_short_history_omits_long_windows(self):
        out = returns_by_calendar(series(list(range(100, 112))))   # ~12 business days
        self.assertIn("1w", out)
        self.assertNotIn("3m", out)

    def test_too_little_data(self):
        self.assertEqual(returns_by_calendar(series([100.0])), {})
        self.assertEqual(returns_by_calendar(pd.Series([], dtype=float)), {})


class Baskets(unittest.TestCase):
    def test_equal_weight_average(self):
        members = [{"1w": 2.0, "1m": 4.0, "3m": 10.0},
                   {"1w": 4.0, "1m": 6.0, "3m": 20.0},
                   {"1w": 6.0, "1m": 2.0, "3m": 30.0}]
        out, used = basket_returns(members)
        self.assertEqual(out, {"1w": 4.0, "1m": 4.0, "3m": 20.0})
        self.assertEqual(used, {"1w": 3, "1m": 3, "3m": 3})

    def test_window_dropped_when_too_few_members(self):
        members = [{"1w": 1.0, "1m": 2.0, "3m": 9.0}, {"1w": 3.0, "1m": 4.0}, {"1w": 5.0, "1m": 6.0}]
        out, used = basket_returns(members)
        self.assertNotIn("3m", out)                  # only 1 member has a 3m return
        self.assertEqual(used["3m"], 1)
        self.assertEqual(out["1w"], 3.0)

    def test_too_few_members_gives_nothing(self):
        out, _ = basket_returns([{"1w": 1.0}] * (MIN_BASKET_MEMBERS - 1))
        self.assertEqual(out, {})


class BenchmarkSelection(unittest.TestCase):
    def test_no_benchmark_is_explicit_not_nifty(self):
        r = get_sector_benchmark("Aviation")
        self.assertFalse(r["available"])
        self.assertEqual(r["kind"], "none")
        self.assertIn("No sector benchmark", r["note"])
        self.assertNotIn("NIFTY 50", str(r["name"]))

    def test_unknown_sector_is_explicit(self):
        r = get_sector_benchmark("Underwater Basket Weaving")
        self.assertFalse(r["available"])
        self.assertIn("not compared against the whole market", r["note"])

    def test_aliases_share_a_benchmark(self):
        self.assertEqual(SECTOR_ALIASES["Oil & Gas"], "Energy")
        self.assertIn("Energy", SECTOR_BENCHMARKS)
        self.assertIn("Infrastructure", SECTOR_BENCHMARKS)


class ConfigIntegrity(unittest.TestCase):
    def test_every_basket_member_is_a_listed_nse_symbol(self):
        rows = _load()[0]
        for sector, cfg in SECTOR_BENCHMARKS.items():
            if cfg["kind"] == "basket":
                self.assertGreaterEqual(len(cfg["symbols"]), MIN_BASKET_MEMBERS, sector)
                for sym in cfg["symbols"]:
                    self.assertIn(sym, rows, f"{sector}: {sym} is not in the NSE list")

    def test_etf_and_index_entries_have_symbols(self):
        for sector, cfg in SECTOR_BENCHMARKS.items():
            self.assertIn(cfg["kind"], ("index", "etf", "basket"), sector)
            if cfg["kind"] in ("index", "etf"):
                self.assertTrue(cfg["symbol"], sector)
            if cfg["kind"] == "etf":
                self.assertTrue(cfg["symbol"].endswith(".NS"), sector)

    def test_index_entries_use_indices_that_have_history(self):
        usable = set(INDEX_SYMBOLS.values())
        for sector, cfg in SECTOR_BENCHMARKS.items():
            if cfg["kind"] == "index":
                self.assertIn(cfg["symbol"], usable, f"{sector} uses an index with no history")

    def test_broken_index_tickers_are_not_used(self):
        broken = {"^CNXFMCG", "^CNXAUTO", "^CNXENERGY", "^CNXMETAL", "^CNXINFRA", "^CNXFIN", "^CNXREALTY"}
        used = {cfg.get("symbol") for cfg in SECTOR_BENCHMARKS.values()} | set(INDEX_SYMBOLS.values())
        self.assertTrue(broken.isdisjoint(used))


class RegimeUnchanged(unittest.TestCase):
    """The regime rule was deliberately left as is; pin its current behaviour."""

    def test_labels(self):
        cases = [({"1m": 5, "3m": 10}, "Risk-On"), ({"1m": 2, "3m": 4}, "Selective Risk-On"),
                 ({"1m": -0.95, "3m": -4.49}, "Range-bound"), ({"1m": -4, "3m": -2}, "Risk-Off"),
                 ({"1m": 0, "3m": -6}, "Mixed")]
        for rets, expected in cases:
            self.assertEqual(classify_market_regime(rets)[0], expected, rets)

    def test_unknown_without_data(self):
        self.assertEqual(classify_market_regime(None)[0], "Unknown")


if __name__ == "__main__":
    unittest.main()
