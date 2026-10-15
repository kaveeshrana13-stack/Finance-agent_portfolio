"""Offline tests for symbol resolution and sector mapping (no network, no LLM)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date

from tools.resolver import (ALIASES, _load, _norm, format_resolution, is_stale,
                            list_downloaded_on, nse_list_size, resolve_stock)
from tools.sector_map import INDUSTRY_TO_SECTOR, SECTOR_TO_SECTOR, map_yahoo_sector
from tools.tickers import clean_ticker_symbol


class CleanTicker(unittest.TestCase):
    def test_keeps_hyphen_and_ampersand(self):
        self.assertEqual(clean_ticker_symbol("BAJAJ-AUTO"), "BAJAJ-AUTO")
        self.assertEqual(clean_ticker_symbol("m&m"), "M&M")

    def test_strips_exchange_suffix_and_junk(self):
        self.assertEqual(clean_ticker_symbol(" tcs.ns "), "TCS")
        self.assertEqual(clean_ticker_symbol("infy.bo"), "INFY")
        self.assertIsNone(clean_ticker_symbol(""))
        self.assertIsNone(clean_ticker_symbol(None))


class ResolveStock(unittest.TestCase):
    def assertResolves(self, query, symbol):
        r = resolve_stock(query)
        self.assertEqual(r["status"], "resolved", f"{query!r} -> {r}")
        self.assertEqual(r["symbol"], symbol, f"{query!r} -> {r}")
        self.assertEqual(r["yahoo_symbol"], symbol + ".NS")

    def test_exact_symbols(self):
        for q, s in [("TCS", "TCS"), ("tcs.ns", "TCS"), ("NSE:INFY", "INFY"),
                     ("M&M", "M&M"), ("BAJAJ-AUTO", "BAJAJ-AUTO"), ("Indigo", "INDIGO")]:
            self.assertResolves(q, s)

    def test_company_names(self):
        for q, s in [("HDFC Bank", "HDFCBANK"), ("Bajaj Auto", "BAJAJ-AUTO"),
                     ("Larsen & Toubro", "LT"), ("InterGlobe Aviation", "INDIGO"),
                     ("Mahindra & Mahindra", "M&M"), ("Tata Consultancy Services", "TCS"),
                     ("Infosys", "INFY"), ("Dr Reddys Laboratories", "DRREDDY")]:
            self.assertResolves(q, s)

    def test_renamed_and_nicknamed(self):
        for q, s in [("Zomato", "ETERNAL"), ("L&T", "LT"), ("SBI", "SBIN"),
                     ("Airtel", "BHARTIARTL"), ("Reliance", "RELIANCE")]:
            self.assertResolves(q, s)

    def test_isin(self):
        rows = _load()[0]
        self.assertResolves(rows["TCS"]["isin"], "TCS")

    def test_demerged_company_is_ambiguous_not_guessed(self):
        r = resolve_stock("Tata Motors")
        self.assertEqual(r["status"], "ambiguous")
        self.assertIsNone(r["symbol"])
        self.assertEqual({c["symbol"] for c in r["candidates"]}, {"TMPV", "TMCV"})

    def test_stale_symbol_is_not_guessed_into_a_company(self):
        """Regression: TATAMOTORS was fuzzy-matched to TMCV and its P/E reported as 'Tata Motors'."""
        r = resolve_stock("TATAMOTORS")
        self.assertEqual(r["status"], "ambiguous")
        self.assertIsNone(r["symbol"])
        self.assertEqual({c["symbol"] for c in r["candidates"]}, {"TMPV", "TMCV"})

    def test_uppercase_typo_asks_but_normal_case_typo_resolves(self):
        upper = resolve_stock("INFOSIS")                  # looks like a ticker -> do not guess
        self.assertEqual(upper["status"], "ambiguous")
        self.assertEqual([c["symbol"] for c in upper["candidates"]], ["INFY"])
        self.assertEqual(resolve_stock("Infosis")["symbol"], "INFY")   # looks like a name -> fuzzy is fine

    def test_generic_prefix_is_ambiguous(self):
        r = resolve_stock("HDFC")
        self.assertEqual(r["status"], "ambiguous")
        self.assertGreater(len(r["candidates"]), 1)

    def test_typo_resolves_with_low_confidence_and_note(self):
        r = resolve_stock("Infosis")
        self.assertEqual(r["status"], "resolved")
        self.assertEqual(r["symbol"], "INFY")
        self.assertEqual(r["match_type"], "fuzzy")
        self.assertIn("confirm", r["note"])

    def test_unknown_is_not_found(self):
        for q in ["Zzzqx Holdings Pvt", "", None]:
            self.assertEqual(resolve_stock(q)["status"], "not_found", q)

    def test_stale_symbol_not_silently_accepted(self):
        # TATAMOTORS and ZOMATO are no longer NSE symbols (renamed/demerged).
        self.assertNotIn("TATAMOTORS", _load()[0])
        self.assertNotIn("ZOMATO", _load()[0])

    def test_format_resolution_is_readable(self):
        self.assertIn("ETERNAL", format_resolution(resolve_stock("Zomato")))
        self.assertIn("ambiguous", format_resolution(resolve_stock("Tata Motors")))
        self.assertIn("not found", format_resolution(resolve_stock("Zzzqx Holdings Pvt")))


class DataIntegrity(unittest.TestCase):
    def test_list_loaded(self):
        self.assertGreater(nse_list_size(), 1000)

    def test_every_alias_points_to_a_real_symbol(self):
        rows = _load()[0]
        for alias, symbols in ALIASES.items():
            for s in symbols:
                self.assertIn(s, rows, f"alias {alias!r} -> {s} is not in the NSE list")

    def test_every_sector_target_is_known_vocabulary(self):
        vocab = {"IT", "Banking", "Pharma", "FMCG", "Auto", "Energy", "Metal", "Infrastructure",
                 "Financial Services", "Cement", "Aviation", "Defence", "Chemicals"}
        for target in list(INDUSTRY_TO_SECTOR.values()) + list(SECTOR_TO_SECTOR.values()):
            self.assertIn(target, vocab)

    def test_norm(self):
        self.assertEqual(_norm("Dr. Reddy's Laboratories Limited"), "dr reddys laboratories")
        self.assertEqual(_norm("Larsen & Toubro Limited"), "larsen and toubro")


class Freshness(unittest.TestCase):
    def test_is_stale(self):
        today = date(2026, 11, 20)
        self.assertFalse(is_stale(date(2026, 10, 25), today=today))   # 26 days old
        self.assertFalse(is_stale(date(2026, 10, 21), today=today))   # exactly 30 days
        self.assertTrue(is_stale(date(2026, 10, 20), today=today))    # 31 days
        self.assertTrue(is_stale(None, today=today))                  # unknown date

    def test_download_date_is_recorded(self):
        self.assertIsNotNone(list_downloaded_on())


class SectorMapping(unittest.TestCase):
    """Cases taken from what Yahoo actually returned for these stocks (2026-10-15)."""

    def test_observed_yahoo_values(self):
        cases = [
            (("Technology", "Information Technology Services"), "IT", True),
            (("Financial Services", "Banks - Regional"), "Banking", True),
            (("Financial Services", "Credit Services"), "Financial Services", True),
            (("Healthcare", "Drug Manufacturers - Specialty & Generic"), "Pharma", True),
            (("Consumer Defensive", "Tobacco"), "FMCG", True),
            (("Consumer Cyclical", "Auto Manufacturers"), "Auto", True),
            (("Energy", "Oil & Gas Refining & Marketing"), "Energy", True),
            (("Basic Materials", "Steel"), "Metal", True),
            (("Basic Materials", "Building Materials"), "Cement", True),
            (("Industrials", "Airlines"), "Aviation", True),
            (("Industrials", "Aerospace & Defense"), "Defence", True),
            (("Industrials", "Engineering & Construction"), "Infrastructure", True),
            # outside the app's vocabulary: kept under Yahoo's name, flagged unmapped
            (("Utilities", "Utilities - Regulated Electric"), "Utilities", False),
            (("Communication Services", "Telecom Services"), "Communication Services", False),
            ((None, None), "Unknown", False),
        ]
        for (sector, industry), expected, mapped in cases:
            got = map_yahoo_sector(sector, industry)
            self.assertEqual(got["sector"], expected, (sector, industry))
            self.assertEqual(got["mapped"], mapped, (sector, industry))


if __name__ == "__main__":
    unittest.main()
