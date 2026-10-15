"""Offline tests for week-over-week memory."""
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.memory import Memory, build_record, diff, summarize_changes


def rec(regime="Range-bound", **stocks):
    return {"regime": regime, "portfolio": {}, "stocks": {s: {"flags": [], "current_price": 100.0, **v}
                                                           for s, v in stocks.items()}}


class Storage(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.mem = Memory(Path(self.dir.name) / "runs.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_first_run_has_no_previous(self):
        self.assertIsNone(self.mem.previous(today=date(2026, 10, 16)))

    def test_previous_is_the_latest_strictly_before_today(self):
        self.mem.save(rec(TCS={"action": "HOLD"}), today=date(2026, 10, 2))
        self.mem.save(rec(TCS={"action": "ADD"}), today=date(2026, 10, 9))
        self.assertEqual(self.mem.previous(today=date(2026, 10, 16))["date"], "2026-10-09")
        self.assertEqual(self.mem.previous(today=date(2026, 10, 9))["date"], "2026-10-02")

    def test_a_second_run_on_the_same_day_replaces_the_first(self):
        self.mem.save(rec(TCS={"action": "HOLD"}), today=date(2026, 10, 16))
        self.mem.save(rec(TCS={"action": "REDUCE"}), today=date(2026, 10, 16))
        runs = self.mem.load()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["stocks"]["TCS"]["action"], "REDUCE")

    def test_a_corrupt_file_is_treated_as_empty(self):
        Path(self.dir.name, "runs.json").write_text("{not json")
        self.assertEqual(self.mem.load(), [])
        self.mem.save(rec(TCS={"action": "HOLD"}), today=date(2026, 10, 16))
        self.assertEqual(len(self.mem.load()), 1)


class Diff(unittest.TestCase):
    def test_first_run_gives_no_changes(self):
        self.assertEqual(diff(None, rec(TCS={"action": "HOLD"})), [])

    def test_action_flag_and_price_changes(self):
        prev = rec(TCS={"action": "HOLD", "flags": ["ROCE_DECLINING"], "current_price": 100.0})
        cur = rec(TCS={"action": "REDUCE", "flags": ["BELOW_200DMA"], "current_price": 92.0})
        kinds = {c["type"]: c for c in diff(prev, cur)}
        self.assertEqual(kinds["action_changed"]["from"], "HOLD")
        self.assertEqual(kinds["action_changed"]["to"], "REDUCE")
        self.assertEqual(kinds["flag_added"]["rule"], "BELOW_200DMA")
        self.assertEqual(kinds["flag_cleared"]["rule"], "ROCE_DECLINING")
        self.assertEqual(kinds["price_move"]["pct"], -8.0)

    def test_small_price_moves_are_not_reported(self):
        prev, cur = rec(TCS={"action": "HOLD"}), rec(TCS={"action": "HOLD", "current_price": 101.0})
        self.assertEqual(diff(prev, cur), [])

    def test_regime_and_portfolio_membership(self):
        prev = rec("Range-bound", TCS={"action": "HOLD"}, ITC={"action": "HOLD"})
        cur = rec("Risk-Off", TCS={"action": "HOLD"}, TITAN={"action": "HOLD"})
        types = sorted(c["type"] for c in diff(prev, cur))
        self.assertEqual(types, ["new_holding", "regime_changed", "removed_holding"])

    def test_summary_text(self):
        changes = diff(rec(TCS={"action": "HOLD"}), rec(TCS={"action": "ADD"}))
        text = summarize_changes(changes, "2026-10-09")
        self.assertIn("TCS changed from HOLD to ADD", text)
        self.assertEqual(summarize_changes([], "2026-10-09"), "")


class Record(unittest.TestCase):
    def test_build_record_from_state(self):
        state = {"regime": {"regime": "Range-bound"},
                 "portfolio": {"total_value": 1000.0, "overall_return_pct": 5.0,
                               "concentration": {"largest_sector": "IT", "largest_sector_pct": 40.0}},
                 "notes": [{"symbol": "TCS", "action": "HOLD"}],
                 "facts": {"TCS": {"snapshot": {"current_price": 2350.0, "roce_pct": 54.9, "pe_ratio": 16.5},
                                   "weight_pct": 40.0}},
                 "flags": {"TCS": [{"rule": "BELOW_200DMA"}, {"rule": "ROCE_DECLINING"}]}}
        r = build_record(state)
        self.assertEqual(r["regime"], "Range-bound")
        self.assertEqual(r["stocks"]["TCS"]["flags"], ["BELOW_200DMA", "ROCE_DECLINING"])
        self.assertEqual(r["stocks"]["TCS"]["current_price"], 2350.0)
        json.dumps(r)                                               # must be serialisable


if __name__ == "__main__":
    unittest.main()
