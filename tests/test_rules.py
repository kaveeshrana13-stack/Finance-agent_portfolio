"""Offline tests for the rulebook, the guardrails and the 200-day average."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.rules import (THRESHOLDS, apply_guardrails, concern_score, evaluate_portfolio, evaluate_stock,
                            top_concerns)
from tools.stocks import earnings_label, moving_average_gap_pct

NA = "not applicable (bank/financial company)"


def ids(flags):
    return {f.rule for f in flags}


class StockRules(unittest.TestCase):
    def test_healthy_stock_has_strengths_and_no_concerns(self):
        flags = evaluate_stock("TCS", {"roce_pct": 54.9, "roce_trend": "improving", "debt_to_equity": 0.1,
                                       "pct_vs_200dma": 6.0, "beta": 0.8, "volatility_pct": 18.0, "fcf_cr": 100.0},
                               weight_pct=8.0, vs_sector_1m=4.0)
        self.assertEqual(concern_score(flags), 0)
        self.assertEqual(ids(flags), {"ROCE_STRONG", "ABOVE_200DMA", "LEADS_SECTOR"})

    def test_weak_stock_collects_concerns(self):
        flags = evaluate_stock("X", {"roce_pct": 9.0, "roce_trend": "declining", "debt_to_equity": 2.5,
                                     "pct_vs_200dma": -8.0, "pct_from_52w_high": -31.0, "fcf_cr": -50.0,
                                     "beta": 1.6, "volatility_pct": 45.0}, weight_pct=5.0, vs_sector_1m=-7.0)
        self.assertTrue({"ROCE_LOW", "ROCE_DECLINING", "LEVERAGE_HIGH", "BELOW_200DMA", "DRAWDOWN", "FCF_NEGATIVE",
                         "LAGS_SECTOR", "BETA_HIGH", "VOLATILITY_HIGH"} <= ids(flags))
        self.assertGreater(concern_score(flags), 10)

    def test_a_bank_is_not_judged_on_roce_or_leverage(self):
        flags = evaluate_stock("HDFCBANK", {"roce_pct": NA, "roce_trend": NA, "debt_to_equity": NA, "fcf_cr": NA,
                                            "pct_vs_200dma": -2.0}, weight_pct=5.0)
        self.assertEqual(ids(flags), {"BELOW_200DMA"})

    def test_missing_values_raise_no_flags(self):
        self.assertEqual(evaluate_stock("X", {}), [])

    def test_concentration_tiers(self):
        self.assertEqual({f.severity for f in evaluate_stock("A", {}, weight_pct=12.0)}, {"watch"})
        self.assertEqual({f.severity for f in evaluate_stock("A", {}, weight_pct=25.0)}, {"high"})
        self.assertEqual(evaluate_stock("A", {}, weight_pct=4.0), [])

    def test_every_flag_carries_its_provenance_and_a_quotable_detail(self):
        flags = evaluate_stock("X", {"roce_pct": 9.0, "pct_vs_200dma": -8.0}, weight_pct=30.0)
        for f in flags:
            self.assertTrue(f.provenance)
            self.assertRegex(f.detail, r"\d")

    def test_thresholds_are_editable(self):
        snap = {"roce_pct": 12.0}
        self.assertIn("ROCE_LOW", ids(evaluate_stock("X", snap)))
        self.assertNotIn("ROCE_LOW", ids(evaluate_stock("X", snap, thresholds={"roce_min_pct": 10.0})))
        self.assertEqual(THRESHOLDS["roce_min_pct"], 15.0)          # the default was not mutated


class SmallPortfolios(unittest.TestCase):
    """Regression: a 5-stock portfolio flagged every position as 'very large'."""

    def sev(self, weight, n):
        return {f.severity for f in evaluate_stock("A", {}, weight_pct=weight, n_holdings=n)}

    def test_an_equal_split_is_not_a_concern(self):
        self.assertEqual(self.sev(20.0, 5), set())            # exactly an equal split of 5
        self.assertEqual(self.sev(26.0, 5), set())            # below 1.5x of an equal split? 26 < 30
        self.assertEqual(self.sev(14.4, 5), set())

    def test_well_above_an_equal_split_is_flagged(self):
        self.assertEqual(self.sev(32.0, 5), {"watch"})
        self.assertEqual(self.sev(45.0, 5), {"high"})

    def test_large_portfolios_keep_the_absolute_limits(self):
        self.assertEqual(self.sev(12.0, 25), {"watch"})
        self.assertEqual(self.sev(25.0, 25), {"high"})
        self.assertEqual(self.sev(6.0, 25), set())

    def test_without_a_holdings_count_nothing_changes(self):
        self.assertEqual(self.sev(25.0, None), {"high"})


class EarningsLabel(unittest.TestCase):
    """Regression: results published 6 days ago were labelled "This week" and raised 'results due soon'."""

    def test_future_and_past(self):
        self.assertEqual(earnings_label(3), "This week")
        self.assertEqual(earnings_label(0), "This week")
        self.assertEqual(earnings_label(10), "Within 2 weeks")
        self.assertEqual(earnings_label(-6), "Recently reported")
        self.assertEqual(earnings_label(-60), "Not soon")
        self.assertEqual(earnings_label(90), "Not soon")

    def test_recent_results_do_not_raise_the_flag(self):
        self.assertEqual(evaluate_stock("TCS", {"earnings_proximity": "Recently reported"}), [])
        self.assertEqual([f.rule for f in evaluate_stock("TCS", {"earnings_proximity": "This week"})], ["EARNINGS_SOON"])


class PortfolioRules(unittest.TestCase):
    def test_sector_tiers(self):
        flags = evaluate_portfolio({"sector_allocation": {"IT": 45.0, "Banking": 27.0, "FMCG": 10.0}})
        by_sector = {f.symbol: f.severity for f in flags}
        self.assertEqual(by_sector, {"IT": "high", "Banking": "watch"})

    def test_balanced_portfolio_is_quiet(self):
        self.assertEqual(evaluate_portfolio({"sector_allocation": {"IT": 20.0, "FMCG": 20.0, "Auto": 20.0}}), [])


class Investigation(unittest.TestCase):
    def test_only_stocks_with_concerns_are_picked_worst_first(self):
        by = {
            "A": evaluate_stock("A", {}, weight_pct=25.0),                                 # high
            "B": evaluate_stock("B", {"roce_pct": 9.0, "pct_vs_200dma": -3.0}),            # two watch
            "C": evaluate_stock("C", {"roce_pct": 40.0, "roce_trend": "improving"}),       # strength only
        }
        self.assertEqual(top_concerns(by, n=2), ["B", "A"])
        self.assertNotIn("C", top_concerns(by, n=5))


class Guardrails(unittest.TestCase):
    def test_add_is_blocked_by_a_high_concentration(self):
        flags = evaluate_stock("A", {}, weight_pct=25.0)
        action, why = apply_guardrails("ADD", flags)
        self.assertEqual(action, "HOLD")
        self.assertIn("rule", why)

    def test_add_is_blocked_by_many_concerns(self):
        flags = evaluate_stock("A", {"roce_pct": 9.0, "roce_trend": "declining", "pct_vs_200dma": -5.0})
        self.assertGreaterEqual(len([f for f in flags if f.kind == "concern"]), 3)
        self.assertEqual(apply_guardrails("ADD", flags)[0], "HOLD")

    def test_guardrails_never_make_a_call_more_aggressive(self):
        flags = evaluate_stock("A", {}, weight_pct=25.0)
        for a in ("HOLD", "REDUCE", "EXIT", "REVIEW"):
            self.assertEqual(apply_guardrails(a, flags), (a, ""))

    def test_clean_add_passes(self):
        self.assertEqual(apply_guardrails("ADD", []), ("ADD", ""))


class MovingAverage(unittest.TestCase):
    def test_gap(self):
        closes = [100.0] * 199 + [110.0]                   # 200 points; average = 100.05
        self.assertAlmostEqual(moving_average_gap_pct(closes, 200), round((110 / 100.05 - 1) * 100, 2), places=2)

    def test_too_short_is_none(self):
        self.assertIsNone(moving_average_gap_pct([100.0] * 50, 200))

    def test_below_average_is_negative(self):
        self.assertLess(moving_average_gap_pct([100.0] * 199 + [80.0], 200), 0)


if __name__ == "__main__":
    unittest.main()
