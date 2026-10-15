"""Offline tests for the answer verifier: realistic ledgers, plus deliberate attacks."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.verify import annotate, extract_numbers, summarize_report, verify_answer


def snap(symbol, **kw):
    base = {"current_price": 2357.9, "pe_ratio": 16.47, "roce_pct": 54.93, "roce_trend": "declining",
            "debt_to_equity": 0.11, "beta": 0.83, "dividend_yield": 2.08, "fcf_cr": 47948.0, "1m_change_pct": 7.94}
    base.update(kw)
    return {"tool": "get_stock_snapshot", "args": {"ticker": symbol},
            "result": {"ok": True, "symbol": symbol, "snapshot": base}}


LEDGER = [
    snap("TITAN", current_price=4455.8, pe_ratio=68.68, **{"1m_change_pct": -9.22}),
    snap("TCS"),
    snap("INFY", current_price=1450.0, pe_ratio=15.12, roce_pct=41.66, roce_trend="improving", beta=0.9),
    {"tool": "get_market_regime", "args": {}, "result": {
        "ok": True, "regime": "Range-bound", "nifty_returns_pct": {"1w": 3.44, "1m": -0.52, "3m": -4.47}}},
    {"tool": "get_stock_news", "args": {"ticker_or_name": "TCS"}, "result": {
        "ok": True, "headlines": [{"date": "2026-10-15", "title": "Buy Tata Consultancy Services; target of Rs 2530: Geojit"}]}},
]


def statuses(answer, ledger=LEDGER):
    r = verify_answer(answer, ledger)
    return r, {(c["claim"], c["status"]) for c in r["claims"]}


class Extraction(unittest.TestCase):
    def raws(self, text):
        return [n["raw"] for n in extract_numbers(text)]

    def test_figures_are_found_with_units(self):
        self.assertEqual(self.raws("P/E 16.47, ROCE 54.93%, FCF ₹47,948 crore"), ["16.47", "54.93%", "₹47,948 crore"])

    def test_non_figures_are_ignored(self):
        text = ("In FY2026 and Q2, over the last 3 months and a 52-week window, 1-month change, "
                "as of 15 Oct 2026, the top 5 holdings.\n1. first\n2. second")
        self.assertEqual(self.raws(text), [])

    def test_lakh_crore(self):
        n = extract_numbers("market cap of ₹1.5 lakh crore")[0]
        self.assertEqual(n["value"], 150000.0)

    def test_negative(self):
        self.assertEqual(extract_numbers("fell to -4.47%")[0]["value"], -4.47)


class Supported(unittest.TestCase):
    def test_exact_figures(self):
        r, s = statuses("TCS has a P/E of 16.47 and ROCE of 54.93%.")
        self.assertEqual(r["verdict"], "verified")
        self.assertEqual(r["n_supported"], 2)

    def test_rounding_is_tolerated_but_only_to_the_stated_precision(self):
        self.assertEqual(statuses("ROCE is about 55%.")[0]["verdict"], "verified")      # 54.93 -> 55
        self.assertEqual(statuses("ROCE is 54.9%.")[0]["verdict"], "verified")
        self.assertEqual(statuses("ROCE is 54.8%.")[0]["verdict"], "unverified")        # not a rounding of 54.93

    def test_currency_commas_and_crore(self):
        self.assertEqual(statuses("Free cash flow was ₹47,948 crore.")[0]["verdict"], "verified")

    def test_sign_agnostic_for_declines(self):
        self.assertEqual(statuses("NIFTY fell 4.47% over three months.")[0]["verdict"], "verified")

    def test_number_quoted_from_a_headline_counts(self):
        r, s = statuses("One broker has a target of Rs 2530.")
        self.assertEqual(r["verdict"], "verified")


class Attacks(unittest.TestCase):
    def test_invented_figure(self):
        r, s = statuses("TCS ROCE is 61.2%.")
        self.assertEqual(r["verdict"], "unverified")
        self.assertIn(("61.2%", "unsupported"), s)

    def test_real_number_under_the_wrong_label(self):
        """54.93 is TCS's ROCE; calling it the P/E must not pass."""
        r, s = statuses("TCS has a P/E of 54.93.")
        self.assertIn(("54.93", "mismatched"), s)
        self.assertEqual(r["verdict"], "unverified")

    def test_partial(self):
        r, _ = statuses("TCS P/E is 16.47 and its beta is 3.1.")
        self.assertEqual(r["verdict"], "partially_verified")
        self.assertEqual(r["n_problems"], 1)

    def test_no_tools_called_means_nothing_is_supported(self):
        r, _ = statuses("TCS P/E is 16.47.", ledger=[])
        self.assertEqual(r["verdict"], "unverified")

    def test_wrong_trend_word(self):
        r, s = statuses("TCS's ROCE is improving.", ledger=[LEDGER[1]])      # tool says declining
        self.assertEqual(r["verdict"], "unverified")
        r, s = statuses("TCS's ROCE is declining.", ledger=[LEDGER[1]])
        self.assertEqual(r["verdict"], "verified")

    def test_trend_claimed_but_never_fetched(self):
        ledger = [{"tool": "get_stock_news", "args": {}, "result": {"ok": True, "headlines": []}}]
        self.assertEqual(statuses("Titan's ROCE is improving.", ledger=ledger)[0]["verdict"], "unverified")

    def test_wrong_regime(self):
        self.assertEqual(statuses("The market regime is Range-bound.")[0]["verdict"], "verified")
        self.assertEqual(statuses("The market regime is Risk-Off.")[0]["verdict"], "unverified")

    def test_risk_on_is_not_confused_with_selective_risk_on(self):
        ledger = [{"tool": "get_market_regime", "args": {}, "result": {"ok": True, "regime": "Selective Risk-On"}}]
        self.assertEqual(statuses("The regime is Selective Risk-On.", ledger=ledger)[0]["verdict"], "verified")


class Derived(unittest.TestCase):
    def test_difference_between_two_stocks_is_derived(self):
        r, s = statuses("TCS's ROCE is 13.27 points higher than Infosys's.")
        claim = next(c for c in r["claims"] if c["claim"] == "13.27")
        self.assertEqual(claim["status"], "derived")
        self.assertEqual(r["verdict"], "verified")

    def test_prices_are_never_combined_into_derived_figures(self):
        """Regression: TITAN price / INFY price = 3.07 once laundered an invented "3" as derived."""
        r, _ = statuses("The ratio is 3.07.")
        self.assertEqual(r["verdict"], "unverified")
        r, _ = statuses("The spread is 3.")
        self.assertEqual(r["verdict"], "unverified")

    def test_a_wrong_difference_is_not_derived(self):
        self.assertEqual(statuses("TCS's ROCE is 14.5 points higher than Infosys's.")[0]["verdict"], "unverified")


class AdversarialTable(unittest.TestCase):
    """Sentences written after the first version missed some and wrongly flagged others."""

    CASES = [
        (True, "TCS (P/E 16.47) trades at ₹2,357.9, up 7.9% in a month, with beta 0.83."),   # was a false alarm
        (True, "Infosys P/E is 15.12 and TCS's is 16.47."),                                  # label carries over
        (True, "Dividend yield of 2.08% and FCF of ₹47,948 crore."),                         # was a false alarm
        (True, "TCS P/E is about 16."),
        (True, "I compared 2 stocks and read 3 headlines."),                                 # counts are not figures
        (True, "Of the 2 stocks, TCS has a P/E of 16.47."),
        (True, "NIFTY gained 3.44% this week."),
        (True, "NIFTY fell 4.47% over three months."),
        (True, "NIFTY's 3-month return is -4.47%."),
        (True, "ROCE of 54.93% for TCS and 41.66% for Infosys."),
        (True, "Titan's price has decreased by 9.22% over the last month."),                 # a % after "price"
        (True, "The price moved down 9.22% in a month."),
        (False, "Titan's price has risen 9.22% over the last month."),                       # it fell
        (False, "TCS's 52-week high is ₹3,500."),                                            # never fetched
        (False, "TCS revenue growth is 2.1%."),                                              # 2.1 ~ dividend 2.08
        (False, "TCS P/E is about 17."),                                                     # 16.47 rounds to 16
        (False, "TCS gained 12.5% this week."),
        (False, "NIFTY rose 4.47% over three months."),                                      # it fell: direction
        (False, "The score is 3."),                                                          # bare integer coincidence
        (False, "TCS ranks 3rd with a ROCE of 54.93%."),                                     # rank never fetched
    ]

    def test_table(self):
        for should_pass, text in self.CASES:
            r = verify_answer(text, LEDGER)
            self.assertEqual(r["n_problems"] == 0, should_pass, f"{text!r} -> {r['verdict']} {r['problems']}")

    def test_direction_reports_what_the_tool_said(self):
        r = verify_answer("NIFTY rose 4.47% over three months.", LEDGER)
        self.assertIn("opposite", r["problems"][0]["source"])
        self.assertIn("-4.47", r["problems"][0]["source"])


class Annotation(unittest.TestCase):
    def test_problem_figures_are_tagged(self):
        answer = "TCS P/E is 16.47 and beta is 3.1."
        out = annotate(answer, verify_answer(answer, LEDGER))
        self.assertIn("3.1 [unverified]", out)
        self.assertNotIn("16.47 [unverified]", out)

    def test_mismatched_label_is_tagged(self):
        answer = "TCS has a P/E of 54.93."
        self.assertIn("54.93 [label mismatch]", annotate(answer, verify_answer(answer, LEDGER)))

    def test_summary_line(self):
        self.assertEqual(summarize_report(verify_answer("No figures here.", LEDGER)), "No numeric claims to verify.")
        self.assertTrue(summarize_report(verify_answer("P/E is 16.47.", LEDGER)).startswith("verified"))


if __name__ == "__main__":
    unittest.main()
