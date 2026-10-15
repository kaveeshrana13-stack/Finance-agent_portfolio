"""Offline tests for the report graph, the model fallback chain and the knowledge base.

No network and no model: tools are a fake toolbox, the model is a scripted fake.
"""
import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("NSE_AUTO_REFRESH", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ledger
from pipeline.graph import ACTIONS, Toolbox, build_graph
from pipeline.knowledge import KnowledgeBase, chunk
from pipeline.llm import AllModelsFailed, FallbackLLM
from tools import resolver

DATA = {
    "TCS": dict(sector="IT", snap={"current_price": 2355.2, "1m_change_pct": 7.94, "pe_ratio": 16.47, "roce_pct": 54.93,
                                   "roce_trend": "declining", "debt_to_equity": 0.11, "beta": 0.83, "volatility_pct": 18.0,
                                   "trend_label": "Weak Uptrend"}),
    "TITAN": dict(sector="Consumer Cyclical", snap={"current_price": 4455.8, "1m_change_pct": -9.22, "pe_ratio": 68.68,
                                                    "roce_pct": 38.43, "roce_trend": "improving", "debt_to_equity": 1.95,
                                                    "beta": 1.05, "volatility_pct": 25.0, "trend_label": "Downtrend"}),
    "HDFCBANK": dict(sector="Banking", snap={"current_price": 724.2, "1m_change_pct": -0.3, "pe_ratio": 15.89,
                                             "roce_pct": "not applicable (bank/financial company)",
                                             "roce_trend": "not applicable (bank/financial company)",
                                             "debt_to_equity": "not applicable (bank/financial company)",
                                             "beta": 1.23, "volatility_pct": 21.0, "trend_label": "Sideways"}),
}
BENCH = {"IT": ("NIFTY IT", 3.94), "Consumer Cyclical": ("Nifty India Consumption ETF (Nippon)", -2.85),
         "Banking": ("NIFTY BANK", 0.03)}


def fake_toolbox(missing=()):
    def snapshot(sym):
        if sym in missing or sym not in DATA:
            return ledger.record("get_stock_snapshot", {"ticker": sym},
                                 {"ok": False, "symbol": sym, "snapshot": {}, "message": "no data"})
        return ledger.record("get_stock_snapshot", {"ticker": sym},
                             {"ok": True, "symbol": sym, "snapshot": dict(DATA[sym]["snap"]), "caveats": ""})

    def benchmark(sector):
        name, one_m = BENCH[sector]
        return ledger.record("get_sector_benchmark", {"sector": sector},
                             {"ok": True, "sector": sector, "kind": "index", "name": name,
                              "returns": {"1w": 1.0, "1m": one_m, "3m": -1.0}})

    def news(sym):
        return ledger.record("get_stock_news", {"ticker_or_name": sym}, {
            "ok": True, "count": 1, "headlines": [{"date": "2026-10-15", "title": f"{sym} wins order worth Rs 500 crore"}]})

    def regime():
        return ledger.record("get_market_regime", {}, {
            "ok": True, "regime": "Range-bound", "nifty_returns_pct": {"1w": 3.44, "1m": -0.52, "3m": -4.47}})

    return Toolbox(resolve=resolver.resolve_stock, snapshot=snapshot, benchmark=benchmark, news=news,
                   regime=regime, sector_of=lambda sym: DATA[sym]["sector"])


class ScriptedLLM:
    """Returns the next scripted reply; an Exception in the script is raised instead."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []
        self.attempts = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r, {"provider": "fake", "model": "scripted", "fallbacks_used": 0}


def notes_json(**overrides):
    base = {
        "TCS": {"action": "HOLD", "rationale": "ROCE is 54.93% but its trend is declining; P/E of 16.47 looks modest.",
                "trigger": "Reduce if the ROCE trend keeps declining."},
        "TITAN": {"action": "REVIEW", "rationale": "Price fell 9.22% in a month against a -2.85% sector move; P/E is 68.68.",
                  "trigger": "Revisit if the ROCE trend turns declining."},
        "HDFCBANK": {"action": "HOLD", "rationale": "ROCE and D/E are not applicable to a bank; P/E is 15.89.",
                     "trigger": "Revisit if sector momentum fades."},
    }
    for sym, patch in overrides.items():
        base[sym] = {**base[sym], **patch}
    return json.dumps({"notes": [{"symbol": s, **v} for s, v in base.items()]})


SUMMARY_OK = "Total value is well spread; the largest holding is a small share and the market regime is Range-bound."
CONCENTRATED = [{"name": "TCS", "quantity": 100, "avg_price": 3000.0},        # ~87% of the book
                {"name": "Titan", "quantity": 5, "avg_price": 3500.0},
                {"name": "HDFC Bank", "quantity": 20, "avg_price": 1500.0}]
HOLDINGS = [{"name": "TCS", "quantity": 10, "avg_price": 3000.0},
            {"name": "Titan", "quantity": 5, "avg_price": 3500.0},
            {"name": "HDFC Bank", "quantity": 20, "avg_price": 1500.0}]


def run(llm, holdings=HOLDINGS, toolbox=None, kb=None, investigate=False, memory=None, today=None, **kw):
    """investigate=False keeps the older tests focused on the notes/summary flow."""
    graph = build_graph(toolbox or fake_toolbox(), llm, kb, today=today, memory=memory, investigate=investigate, **kw)
    return graph.invoke({"holdings": holdings})


class HappyPath(unittest.TestCase):
    def test_two_model_calls_and_everything_verified(self):
        llm = ScriptedLLM([notes_json(), SUMMARY_OK])
        out = run(llm)
        self.assertEqual(len(llm.prompts), 2)                       # notes + summary: that is all
        self.assertTrue(all(n["verified"] for n in out["notes"]))
        self.assertEqual({n["symbol"] for n in out["notes"]}, {"TCS", "TITAN", "HDFCBANK"})
        self.assertIn("Weekly portfolio report", out["report_md"])
        self.assertIn("✓ verified", out["report_md"])
        self.assertNotIn("contains unverified figures", out["report_md"])

    def test_sector_comparison_is_computed_by_code(self):
        out = run(ScriptedLLM([notes_json(), SUMMARY_OK]))
        self.assertEqual(out["facts"]["TCS"]["vs_sector_1m"], 4.0)             # 7.94 - 3.94
        self.assertEqual(out["facts"]["TITAN"]["vs_sector_1m"], -6.37)         # -9.22 - -2.85

    def test_actions_are_normalised(self):
        out = run(ScriptedLLM([notes_json(TCS={"action": "strong buy!!"}), SUMMARY_OK]))
        tcs = next(n for n in out["notes"] if n["symbol"] == "TCS")
        self.assertIn(tcs["action"], ACTIONS)
        self.assertEqual(tcs["action"], "REVIEW")

    def test_portfolio_figures_come_from_code(self):
        out = run(ScriptedLLM([notes_json(), SUMMARY_OK]))
        p = out["portfolio"]
        self.assertEqual(p["total_value"], round(10 * 2355.2 + 5 * 4455.8 + 20 * 724.2, 2))
        self.assertEqual(sum(p["sector_allocation"].values()), 100.0, p["sector_allocation"])

    def test_bank_is_not_given_a_roce(self):
        out = run(ScriptedLLM([notes_json(), SUMMARY_OK]))
        self.assertIn("n/a (financial)", out["report_md"])


class VerificationLoop(unittest.TestCase):
    def test_invented_figure_is_fixed_on_retry(self):
        bad = notes_json(TCS={"rationale": "ROCE is 61.2% and falling."})
        llm = ScriptedLLM([bad, notes_json(), SUMMARY_OK])
        out = run(llm)
        self.assertEqual(len(llm.prompts), 3)                                   # notes, notes again, summary
        self.assertIn("61.2", llm.prompts[1])                                   # told exactly what was wrong
        self.assertTrue(all(n["verified"] for n in out["notes"]))
        self.assertNotIn("61.2", out["report_md"])

    def test_persistent_invention_is_tagged_not_hidden(self):
        bad = notes_json(TCS={"rationale": "ROCE is 61.2% and falling."})
        llm = ScriptedLLM([bad, bad, SUMMARY_OK])
        out = run(llm)
        self.assertEqual(len(llm.prompts), 3)                                   # capped: one retry only
        tcs = next(n for n in out["notes"] if n["symbol"] == "TCS")
        self.assertFalse(tcs["verified"])
        self.assertIn("61.2% [unverified]", tcs["rationale"])
        self.assertIn("contains unverified figures", out["report_md"])

    def test_invented_threshold_in_a_trigger_is_caught(self):
        bad = notes_json(TITAN={"trigger": "Exit if the price falls below 3800."})
        out = run(ScriptedLLM([bad, bad, SUMMARY_OK]))
        titan = next(n for n in out["notes"] if n["symbol"] == "TITAN")
        self.assertFalse(titan["verified"])

    def test_wrong_figure_in_summary_is_retried(self):
        llm = ScriptedLLM([notes_json(), "Overall return is 76.5% and the book is concentrated.", SUMMARY_OK])
        out = run(llm)
        self.assertEqual(len(llm.prompts), 3)
        self.assertEqual(out["summary_problems"], [])
        self.assertNotIn("76.5", out["report_md"])

    def test_correct_portfolio_figure_in_summary_passes(self):
        total = round(10 * 2355.2 + 5 * 4455.8 + 20 * 724.2, 2)
        llm = ScriptedLLM([notes_json(), f"Total value is ₹{total:,.0f}; regime is Range-bound."])
        out = run(llm)
        self.assertEqual(len(llm.prompts), 2)
        self.assertEqual(out["summary_problems"], [])


class PromptsAndStyle(unittest.TestCase):
    def test_internal_tags_never_reach_the_report(self):
        """Regression: a note said "the cause is unknown [no evidence]", echoing an internal marker."""
        bad = notes_json(TCS={"rationale": "ROCE is 54.93% and its cause is unknown [no evidence]."})
        out = run(ScriptedLLM([bad, SUMMARY_OK]))
        tcs = next(n for n in out["notes"] if n["symbol"] == "TCS")
        self.assertNotIn("no evidence", tcs["rationale"])
        self.assertTrue(tcs["verified"])
        self.assertNotIn("[no evidence]", out["report_md"])

    def test_facts_are_shown_as_plain_labels_not_field_names(self):
        llm = ScriptedLLM([notes_json(), SUMMARY_OK])
        run(llm)
        prompt = llm.prompts[0]
        self.assertIn("P/E ratio: 16.47", prompt)
        self.assertIn("1-month price change: 7.94%", prompt)
        self.assertIn("percentage points ahead of the benchmark", prompt)
        for leak in ("pe_ratio", "roce_pct", "1m_change_pct", "vs_sector_1m_pct"):
            self.assertNotIn(leak, prompt)

    def test_field_names_in_a_note_trigger_one_retry(self):
        leaky = notes_json(TCS={"rationale": "Its pe_ratio of 16.47 and roce_pct of 54.93 look fine."})
        llm = ScriptedLLM([leaky, notes_json(), SUMMARY_OK])
        out = run(llm)
        self.assertEqual(len(llm.prompts), 3)
        self.assertIn("internal field names", llm.prompts[1])
        self.assertTrue(all(n["verified"] for n in out["notes"]))

    def test_a_trigger_that_is_not_a_condition_is_a_problem(self):
        bad = notes_json(TCS={"trigger": "pe_ratio"})
        llm = ScriptedLLM([bad, bad, SUMMARY_OK])
        out = run(llm)
        tcs = next(n for n in out["notes"] if n["symbol"] == "TCS")
        self.assertFalse(tcs["verified"])

    def test_prompts_demand_a_change_from_the_current_state_and_a_specific_watch_item(self):
        llm = ScriptedLLM([notes_json(), SUMMARY_OK])
        run(llm)
        self.assertIn("CHANGE from the current state", llm.prompts[0])
        self.assertIn("never \"the decisions\"", llm.prompts[1])

    def test_summary_facts_are_plain_text(self):
        llm = ScriptedLLM([notes_json(), SUMMARY_OK])
        run(llm)
        self.assertIn("Total portfolio value: ₹", llm.prompts[1])
        self.assertNotIn("total_value", llm.prompts[1])


class Degradation(unittest.TestCase):
    def test_no_model_available_still_gives_a_report(self):
        llm = ScriptedLLM([AllModelsFailed([{"provider": "gemini", "model": "x", "error": "429"}])])
        out = run(llm)
        self.assertIn("Analysis was not generated", out["report_md"])
        self.assertIn("Holdings", out["report_md"])
        self.assertTrue(any("no model answered" in w for w in out["warnings"]))
        self.assertEqual(len(llm.prompts), 1)                                   # no second call after total failure

    def test_data_only_mode_makes_no_model_calls(self):
        out = run(None)
        self.assertIn("Range-bound", out["report_md"])
        self.assertIn("Analysis was not generated", out["report_md"])
        self.assertIn("none (data-only run)", out["report_md"])

    def test_garbage_json_becomes_review_notes(self):
        out = run(ScriptedLLM(["I am sorry, I cannot do that.", SUMMARY_OK]))
        self.assertTrue(all(n["action"] == "REVIEW" for n in out["notes"]))
        self.assertIn("returned no note", out["notes"][0]["rationale"])

    def test_fenced_json_is_accepted(self):
        out = run(ScriptedLLM(["```json\n" + notes_json() + "\n```", SUMMARY_OK]))
        self.assertEqual(next(n for n in out["notes"] if n["symbol"] == "TCS")["action"], "HOLD")

    def test_ambiguous_and_unknown_holdings_are_skipped_with_a_reason(self):
        holdings = HOLDINGS + [{"name": "Tata Motors", "quantity": 1, "avg_price": 1.0},
                               {"name": "Zzzqx Holdings Pvt", "quantity": 1, "avg_price": 1.0}]
        out = run(ScriptedLLM([notes_json(), SUMMARY_OK]), holdings=holdings)
        text = " ".join(out["warnings"])
        self.assertIn("Tata Motors", text)
        self.assertIn("TMPV", text)
        self.assertIn("Zzzqx", text)
        self.assertEqual({n["symbol"] for n in out["notes"]}, {"TCS", "TITAN", "HDFCBANK"})

    def test_a_holding_with_no_market_data_is_reported_not_fatal(self):
        out = run(ScriptedLLM([notes_json(), SUMMARY_OK]), toolbox=fake_toolbox(missing=("TITAN",)))
        self.assertTrue(any("TITAN" in w and "no market data" in w for w in out["warnings"]))
        self.assertNotIn("TITAN", {n["symbol"] for n in out["notes"]})


class FakeKB:
    def __init__(self, passages):
        self.passages, self.queries = passages, []

    def __len__(self):
        return len(self.passages)

    def search(self, query, k=3, symbol=None):
        self.queries.append((query, symbol))
        return self.passages[:k]


def act(action, **kw):
    return json.dumps({"action": action, **kw})


TRIAGE_TCS = json.dumps({"investigate": [{"symbol": "TCS", "question": "Why is ROCE declining?"}]})
TCS_DOC = {"source": "call.pdf", "page": 8, "text": "For FY26 our operating margin was 25 percent, up 70 basis points."}


class ResearchAgent(unittest.TestCase):
    def script(self, finding_claim="Operating margin was 25 percent, up 70 basis points.", **notes):
        return [TRIAGE_TCS,
                act("search_documents", args={"query": "operating margin"}, why="margins drive ROCE"),
                act("finish", findings=[{"claim": finding_claim, "evidence": ["E1"]}]),
                notes_json(**notes), SUMMARY_OK]

    def test_the_model_chooses_what_to_look_into_and_the_findings_reach_the_notes(self):
        kb = FakeKB([TCS_DOC])
        llm = ScriptedLLM(self.script())
        out = run(llm, kb=kb, investigate=True, max_stocks=1, max_tool_calls=1)
        self.assertEqual(len(llm.prompts), 5)                       # triage + 2 agent steps + notes + summary
        self.assertEqual(out["triage"]["picks"][0]["symbol"], "TCS")
        self.assertEqual(kb.queries, [("operating margin", "TCS")])  # the model's own query, scoped to the company
        self.assertIn("Research finding [E1]", llm.prompts[3])
        self.assertIn("Operating margin was 25 percent", llm.prompts[3])
        self.assertIn("## Research agent", out["report_md"])
        self.assertIn("call.pdf, p.8", out["report_md"])
        self.assertEqual(out["figures"]["research"]["problems"], 0)

    def test_an_invented_finding_is_marked_in_the_report(self):
        llm = ScriptedLLM(self.script(finding_claim="Operating margin was 31 percent, up 70 basis points."))
        out = run(llm, kb=FakeKB([TCS_DOC]), investigate=True, max_stocks=1, max_tool_calls=1)
        self.assertEqual(out["figures"]["research"]["problems"], 1)
        self.assertIn("⚠", out["report_md"].split("## Research agent")[1])

    def test_the_agent_is_skipped_when_asked(self):
        llm = ScriptedLLM([notes_json(), SUMMARY_OK])
        out = run(llm, investigate=False)
        self.assertEqual(len(llm.prompts), 2)
        self.assertEqual(out["investigations"], [])

    def test_a_total_model_outage_during_research_still_gives_a_report(self):
        llm = ScriptedLLM([AllModelsFailed([])] * 8)
        out = run(llm, kb=FakeKB([TCS_DOC]), investigate=True)
        self.assertIn("Weekly portfolio report", out["report_md"])


class RulesAndGuardrails(unittest.TestCase):
    def test_flags_are_computed_and_reach_the_prompt(self):
        llm = ScriptedLLM([notes_json(), SUMMARY_OK])
        out = run(llm, holdings=CONCENTRATED)
        rules = {f["rule"] for f in out["flags"]["TCS"]}
        self.assertIn("ROCE_DECLINING", rules)
        self.assertIn("STOCK_CONCENTRATION", rules)                  # TCS is ~87% of this portfolio
        self.assertIn("Rule flag (concern", llm.prompts[0])
        self.assertIn("PORTFOLIO FLAG", llm.prompts[0])               # IT is the biggest sector

    def test_add_is_blocked_on_an_over_concentrated_position(self):
        out = run(ScriptedLLM([notes_json(TCS={"action": "ADD"}), SUMMARY_OK]), holdings=CONCENTRATED)
        tcs = next(n for n in out["notes"] if n["symbol"] == "TCS")
        self.assertEqual(tcs["action"], "HOLD")
        self.assertEqual(tcs["model_action"], "ADD")
        self.assertEqual(out["guardrails"][0]["symbol"], "TCS")
        self.assertIn("Guardrail", out["report_md"])

    def test_guardrails_never_turn_a_cautious_call_into_a_bolder_one(self):
        out = run(ScriptedLLM([notes_json(TCS={"action": "REDUCE"}), SUMMARY_OK]), holdings=CONCENTRATED)
        self.assertEqual(next(n for n in out["notes"] if n["symbol"] == "TCS")["action"], "REDUCE")
        self.assertEqual(out["guardrails"], [])

    def test_a_figure_from_a_flag_detail_is_a_supported_fact(self):
        text = {"rationale": "TCS is 86.5% of the portfolio, more than twice an equal split, so size matters here."}
        out = run(ScriptedLLM([notes_json(TCS=text), SUMMARY_OK]), holdings=CONCENTRATED)
        self.assertTrue(next(n for n in out["notes"] if n["symbol"] == "TCS")["verified"])


class WeekOverWeek(unittest.TestCase):
    def test_second_run_reports_what_changed(self):
        import tempfile
        from datetime import date
        from pipeline.memory import Memory
        with tempfile.TemporaryDirectory() as d:
            mem = Memory(Path(d) / "runs.json")
            run(ScriptedLLM([notes_json(), SUMMARY_OK]), memory=mem, today=date(2026, 10, 9))
            second = run(ScriptedLLM([notes_json(TITAN={"action": "REDUCE"}), SUMMARY_OK]), memory=mem,
                         today=date(2026, 10, 16))
            self.assertEqual(second["previous"]["date"], "2026-10-09")
            kinds = {(c["type"], c["symbol"]) for c in second["changes"]}
            self.assertIn(("action_changed", "TITAN"), kinds)
            self.assertIn("## What changed since 2026-10-09", second["report_md"])
            self.assertIn("TITAN changed from REVIEW to REDUCE", second["report_md"])
            self.assertEqual(len(mem.load()), 2)

    def test_first_run_has_no_comparison_section(self):
        import tempfile
        from pipeline.memory import Memory
        with tempfile.TemporaryDirectory() as d:
            out = run(ScriptedLLM([notes_json(), SUMMARY_OK]), memory=Memory(Path(d) / "r.json"))
            self.assertNotIn("What changed since", out["report_md"])


class Fallback(unittest.TestCase):
    CHAIN = [("gemini", "a"), ("gemini", "b"), ("ollama", "c")]

    def test_first_working_model_answers(self):
        llm = FallbackLLM(self.CHAIN, complete=lambda p, m, msgs, t, temp: f"{p}/{m}")
        text, meta = llm.generate("hi")
        self.assertEqual((text, meta["fallbacks_used"]), ("gemini/a", 0))

    def test_quota_error_falls_through_and_the_model_is_skipped_afterwards(self):
        calls = []

        def complete(provider, model, msgs, timeout, temp):
            calls.append(model)
            if model == "a":
                raise RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")
            return f"{provider}/{model}"
        llm = FallbackLLM(self.CHAIN, complete=complete)
        self.assertEqual(llm.generate("1")[0], "gemini/b")
        self.assertEqual(llm.generate("2")[0], "gemini/b")
        self.assertEqual(calls, ["a", "b", "b"])                               # "a" was not retried

    def test_every_model_failing_raises_with_the_reasons(self):
        def complete(*a):
            raise RuntimeError("down")
        with self.assertRaises(AllModelsFailed) as cm:
            FallbackLLM(self.CHAIN, complete=complete).generate("x")
        self.assertEqual(len(cm.exception.attempts), 3)

    def test_the_key_is_never_leaked_into_error_text(self):
        os.environ["GEMINI_API_KEY"] = "AIzaSECRETSECRETSECRET"
        try:
            def complete(*a):
                raise RuntimeError("auth failed for AIzaSECRETSECRETSECRET")
            llm = FallbackLLM([("gemini", "a")], complete=complete)
            with self.assertRaises(AllModelsFailed):
                llm.generate("x")
            self.assertNotIn("AIzaSECRET", json.dumps(llm.attempts))
        finally:
            os.environ.pop("GEMINI_API_KEY", None)


class Knowledge(unittest.TestCase):
    def test_ranking_and_citation_fields(self):
        kb = KnowledgeBase([{"source": "a.txt", "page": 1, "text": "gold hedging improved jewellery margins"},
                            {"source": "b.txt", "page": 2, "text": "data centre capacity expansion"}])
        hit = kb.search("jewellery margins")[0]
        self.assertEqual((hit["source"], hit["page"]), ("a.txt", 1))
        self.assertEqual(kb.search("banana"), [])

    def test_small_corpus_still_scores(self):
        """Regression: textbook BM25 gave a term in half the documents a weight of exactly 0."""
        kb = KnowledgeBase([{"source": "a.txt", "page": None, "text": "alpha beta"},
                            {"source": "b.txt", "page": None, "text": "gamma delta"}])
        self.assertEqual(kb.search("alpha")[0]["source"], "a.txt")

    def test_readme_files_are_not_indexed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            Path(d, "README.md").write_text("Drop annual reports here")
            Path(d, "titan.txt").write_text("Titan jewellery margins improved")
            kb = KnowledgeBase.from_directory(d)
        self.assertEqual({p["source"] for p in kb.passages}, {"titan.txt"})

    def test_empty_knowledge_base(self):
        self.assertEqual(KnowledgeBase([]).search("anything"), [])

    def test_chunking_overlaps(self):
        words = " ".join(str(i) for i in range(300))
        chunks = chunk(words, size=100, overlap=20)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[1].split()[0], "80")


if __name__ == "__main__":
    unittest.main()
