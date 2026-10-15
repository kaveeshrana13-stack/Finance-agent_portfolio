"""Offline tests for the news tool (no network)."""
import sys
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.news as N
from tools.news import build_queries, build_query, parse_feed, search_terms, select_items

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def rss(*items):
    body = "".join(
        f"<item><title>{t}</title><link>https://news.google.com/rss/articles/{i}</link>"
        f"<pubDate>{d}</pubDate></item>" for i, (t, d) in enumerate(items)
    )
    return f"<?xml version='1.0'?><rss><channel>{body}</channel></rss>".encode()


class SearchTerms(unittest.TestCase):
    def test_name_and_symbol(self):
        self.assertEqual(search_terms("Infosys Limited", "INFY"), ["Infosys", "INFY"])
        self.assertEqual(search_terms("Tata Consultancy Services Limited", "TCS"), ["Tata Consultancy Services", "TCS"])

    def test_one_and_two_letter_symbols_are_dropped(self):
        self.assertEqual(search_terms("Larsen & Toubro Limited", "LT"), ["Larsen & Toubro"])
        self.assertEqual(search_terms("Bharat Electronics Limited", "BEL"), ["Bharat Electronics", "BEL"])

    def test_no_duplicates(self):
        self.assertEqual(search_terms("Titan Company Limited", "TITAN"), ["Titan"])

    def test_query_shape(self):
        self.assertEqual(build_query(["Infosys", "INFY"]),
                         "(intitle:Infosys OR intitle:INFY) site:moneycontrol.com/news when:7d")
        self.assertEqual(build_query(["KPI Green Energy"], days=3),
                         'intitle:"KPI Green Energy" site:moneycontrol.com/news when:3d')


class Ambiguous(unittest.TestCase):
    def test_itc_excludes_input_tax_credit_stories(self):
        q = build_query(["ITC"])
        self.assertIn('-"input tax credit"', q)
        self.assertIn("-GST", q)
        self.assertNotIn("-GST", build_query(["Infosys"]))


class MultiQuery(unittest.TestCase):
    def test_one_query_per_term(self):
        qs = build_queries(["Tata Consultancy Services", "TCS"])
        self.assertEqual(qs, ['intitle:"Tata Consultancy Services" site:moneycontrol.com/news when:7d',
                              "intitle:TCS site:moneycontrol.com/news when:7d"])

    def test_results_from_every_query_are_merged_and_deduped(self):
        feeds = {
            'intitle:"Tata Consultancy Services" site:moneycontrol.com/news when:7d':
                rss(("TCS wins deal - Moneycontrol.com", "Wed, 14 Oct 2026 09:00:00 GMT")),
            "intitle:TCS site:moneycontrol.com/news when:7d":
                rss(("TCS wins deal - Moneycontrol.com", "Wed, 14 Oct 2026 09:00:00 GMT"),
                    ("TCS Q2 preview - Moneycontrol.com", "Thu, 15 Oct 2026 08:00:00 GMT")),
        }
        with unittest.mock.patch.object(N, "_fetch_feed", lambda q, timeout=15: feeds[q]), \
             unittest.mock.patch.object(N, "datetime") as dt:
            dt.now.side_effect = lambda tz=None: NOW if tz else NOW.replace(tzinfo=None)
            r = N.get_stock_news("TCS")
        self.assertEqual(r["status"], "ok")
        self.assertEqual([i["title"] for i in r["data"]["items"]], ["TCS Q2 preview", "TCS wins deal"])

    def test_one_failed_query_is_partial_not_fatal(self):
        def fetch(q, timeout=15):
            if "Tata Consultancy" in q:
                raise N.urllib.error.URLError("boom")
            return rss(("TCS Q2 preview - Moneycontrol.com", "Thu, 15 Oct 2026 08:00:00 GMT"))
        with unittest.mock.patch.object(N, "_fetch_feed", fetch), unittest.mock.patch.object(N, "datetime") as dt:
            dt.now.side_effect = lambda tz=None: NOW if tz else NOW.replace(tzinfo=None)
            r = N.get_stock_news("TCS")
        self.assertEqual(r["status"], "partial")
        self.assertEqual(r["data"]["count"], 1)
        self.assertTrue(any("failed" in w for w in r["warnings"]))

    def test_all_queries_failing_is_an_error(self):
        def fetch(q, timeout=15):
            raise N.urllib.error.URLError("down")
        with unittest.mock.patch.object(N, "_fetch_feed", fetch):
            r = N.get_stock_news("TCS")
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["data"]["count"], 0)


class ParseAndSelect(unittest.TestCase):
    def feed(self):
        return rss(
            ("Infosys wins mega deal - Moneycontrol.com", "Wed, 14 Oct 2026 09:00:00 GMT"),
            ("Infosys wins mega deal - Moneycontrol.com", "Wed, 14 Oct 2026 09:05:00 GMT"),   # duplicate title
            ("Infosys Q2 preview - Moneycontrol.com", "Thu, 15 Oct 2026 08:00:00 GMT"),
            ("Old Infosys story - Moneycontrol.com", "Fri, 18 Aug 2023 10:00:00 GMT"),        # outside window
            ("Business News, Economic News, Indian Stock Market News - Moneycontrol.com", "Thu, 15 Oct 2026 07:00:00 GMT"),
            ("Share/Stock Market News - Latest NSE, BSE, Business News - Moneycontrol.com", "Thu, 15 Oct 2026 07:00:00 GMT"),
            ("No date at all", ""),
        )

    def test_parse_strips_suffix_and_skips_generic_and_undated(self):
        titles = [i["title"] for i in parse_feed(self.feed())]
        self.assertIn("Infosys Q2 preview", titles)
        self.assertNotIn("No date at all", titles)
        self.assertFalse(any("Business News" in t or "Latest NSE" in t for t in titles))
        self.assertFalse(any(t.endswith("Moneycontrol.com") for t in titles))

    def test_select_window_dedupe_and_order(self):
        out = select_items(parse_feed(self.feed()), days=7, now=NOW)
        self.assertEqual([i["title"] for i in out], ["Infosys Q2 preview", "Infosys wins mega deal"])
        self.assertTrue(out[0]["date"] >= out[1]["date"])           # newest first

    def test_max_items(self):
        many = rss(*[(f"Headline {n}", "Wed, 14 Oct 2026 09:00:00 GMT") for n in range(30)])
        self.assertEqual(len(select_items(parse_feed(many), days=7, max_items=10, now=NOW)), 10)

    def test_future_dated_items_are_dropped(self):
        future = rss(("From the future", (NOW + timedelta(days=3)).strftime("%a, %d %b %Y %H:%M:%S GMT")))
        self.assertEqual(select_items(parse_feed(future), days=7, now=NOW), [])

    def test_bare_outlet_title_is_not_a_headline(self):
        feed = rss(("- Moneycontrol.com", "Wed, 14 Oct 2026 09:00:00 GMT"), ("Real headline - Moneycontrol.com", "Wed, 14 Oct 2026 09:00:00 GMT"))
        self.assertEqual([i["title"] for i in parse_feed(feed)], ["Real headline"])

    def test_empty_feed(self):
        self.assertEqual(select_items(parse_feed(rss()), now=NOW), [])


if __name__ == "__main__":
    unittest.main()
