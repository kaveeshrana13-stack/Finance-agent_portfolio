"""Offline tests for the research agent: triage, the bounded loop and the checks on its findings."""
import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("NSE_AUTO_REFRESH", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ledger
from pipeline.investigator import (Tools, check_findings, default_question, investigate_stock, parse_action,
                                   parse_triage, triage)
from pipeline.llm import AllModelsFailed
from pipeline.rules import evaluate_stock


class Scripted:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r, {"provider": "fake", "model": "scripted"}


def act(action, **kw):
    return json.dumps({"action": action, **kw})


FLAGS = evaluate_stock("TCS", {"roce_pct": 54.9, "roce_trend": "declining", "pct_vs_200dma": -4.0}, weight_pct=12.0)
DOC = {"source": "call.pdf", "page": 8, "text": "For FY26 our operating margin was 25%, an expansion of 70 basis points."}


def tools(docs=None, **kw):
    return Tools(docs=docs or (lambda q, s: [DOC]), news=kw.get("news"), fundamentals=kw.get("fundamentals"),
                 risk=kw.get("risk"))


class Parsing(unittest.TestCase):
    def test_tool_call(self):
        a = parse_action(act("search_documents", args={"query": "margin"}, why="check margins"))
        self.assertEqual((a["action"], a["args"]), ("search_documents", {"query": "margin"}))

    def test_finish_with_fenced_json(self):
        reply = "```json\n" + act("finish", findings=[{"claim": "x", "evidence": ["E1"]}]) + "\n```"
        self.assertEqual(parse_action(reply)["findings"][0]["claim"], "x")

    def test_unknown_tool_and_garbage_are_rejected(self):
        self.assertIsNone(parse_action(act("delete_everything")))
        self.assertIsNone(parse_action("I think we should look at the news."))
        self.assertIsNone(parse_action(""))

    def test_triage_parsing(self):
        reply = json.dumps({"investigate": [{"symbol": "tcs", "question": "why?"}, {"symbol": "TCS", "question": "dup"},
                                            {"symbol": "ZZZ", "question": "not a candidate"},
                                            {"symbol": "INFY", "question": "also?"}, {"symbol": "ITC", "question": "third"}]})
        picks = parse_triage(reply, {"TCS", "INFY", "ITC"}, max_stocks=2)
        self.assertEqual([p["symbol"] for p in picks], ["TCS", "INFY"])


class Triage(unittest.TestCase):
    FLAGS = {"A": evaluate_stock("A", {"roce_pct": 9.0}, weight_pct=25.0),
             "B": evaluate_stock("B", {"roce_pct": 9.0, "pct_vs_200dma": -3.0}),
             "C": evaluate_stock("C", {"roce_pct": 40.0, "roce_trend": "improving"})}

    def test_the_model_chooses_and_writes_the_question(self):
        llm = Scripted([json.dumps({"investigate": [{"symbol": "B", "question": "Is the weak ROCE temporary?"}]})])
        picks, n, how = triage(llm, self.FLAGS, {"A": "A Ltd", "B": "B Ltd"}, max_stocks=2)
        self.assertEqual(picks, [{"symbol": "B", "question": "Is the weak ROCE temporary?"}])
        self.assertEqual((n, how), (1, "model"))
        self.assertNotIn("C ", llm.prompts[0].split("Choose")[0].replace("C (", "").replace("C:", ""))

    def test_unusable_reply_falls_back_to_rule_scores(self):
        picks, n, how = triage(Scripted(["no idea"]), self.FLAGS, {}, max_stocks=2)
        self.assertEqual([p["symbol"] for p in picks], ["A", "B"])   # 5 (weak ROCE + 25% position) vs 4
        self.assertIn("rule scores", how)
        self.assertTrue(all(p["question"] for p in picks))

    def test_model_outage_falls_back_too(self):
        picks, _, how = triage(Scripted([AllModelsFailed([])]), self.FLAGS, {}, max_stocks=1)
        self.assertEqual(len(picks), 1)
        self.assertIn("rule scores", how)

    def test_nothing_flagged_costs_no_request(self):
        llm = Scripted([])
        picks, n, how = triage(llm, {"C": self.FLAGS["C"]}, {}, max_stocks=2)
        self.assertEqual((picks, n, llm.prompts), ([], 0, []))

    def test_default_question_names_the_concerns(self):
        self.assertIn("declining", default_question("TCS", FLAGS).lower())


class Loop(unittest.TestCase):
    def setUp(self):
        ledger.reset()

    def run_loop(self, replies, max_tool_calls=2, **kw):
        llm = Scripted(replies)
        r = investigate_stock(llm, kw.pop("tools", tools()), "TCS", "Tata Consultancy Services", "IT",
                              "Why is ROCE declining?", FLAGS, max_tool_calls=max_tool_calls,
                              doc_sources=kw.pop("doc_sources", ["call.pdf"]))
        return r, llm

    def test_the_model_picks_a_tool_then_finishes_with_cited_findings(self):
        r, llm = self.run_loop([
            act("search_documents", args={"query": "operating margin"}, why="margins explain ROCE"),
            act("finish", findings=[{"claim": "Operating margin was 25%, up 70 basis points.", "evidence": ["E1"]}])])
        self.assertEqual([s["action"] for s in r["steps"]], ["search_documents", "finish"])
        self.assertEqual(r["steps"][0]["args"], {"query": "operating margin"})
        self.assertEqual(r["evidence"][0]["source"], "call.pdf")
        self.assertEqual((r["requests"], r["tool_calls"], r["stopped"]), (2, 1, "finished"))
        self.assertTrue(r["findings"][0]["verified"], r["findings"])
        self.assertIn("[E1]", llm.prompts[1])                       # the second step sees the evidence

    def test_the_budget_is_a_hard_cap(self):
        asks = [act("get_news") for _ in range(10)]
        r, llm = self.run_loop(asks, max_tool_calls=2, tools=tools(news=lambda s: {"headlines": []}))
        self.assertEqual(r["tool_calls"], 2)
        self.assertLessEqual(r["requests"], 3)                      # 2 tool calls + 1 forced close
        self.assertEqual(r["stopped"], "finished")
        self.assertIn("MUST finish", llm.prompts[2])

    def test_it_may_stop_early(self):
        r, _ = self.run_loop([act("finish", findings=[])], max_tool_calls=3)
        self.assertEqual((r["requests"], r["tool_calls"]), (1, 0))

    def test_unusable_replies_cannot_loop_forever(self):
        r, _ = self.run_loop(["??"] * 20, max_tool_calls=2)
        self.assertEqual(r["stopped"], "unusable_replies")
        self.assertLessEqual(r["requests"], 4)

    def test_model_outage_is_reported(self):
        r, _ = self.run_loop([act("get_news"), AllModelsFailed([])], tools=tools(news=lambda s: {"headlines": []}))
        self.assertEqual(r["stopped"], "model_unavailable")
        self.assertEqual(r["findings"], [])

    def test_a_failing_tool_does_not_end_the_run(self):
        def boom(symbol):
            raise RuntimeError("down")
        r, _ = self.run_loop([act("get_risk_profile"), act("finish", findings=[])], tools=tools(risk=boom))
        self.assertIn("failed", r["evidence"][0]["text"])
        self.assertEqual(r["stopped"], "finished")

    def test_a_tool_that_is_not_wired_in_says_so(self):
        r, _ = self.run_loop([act("get_fundamentals_history"), act("finish", findings=[])], tools=Tools())
        self.assertIn("not available", r["evidence"][0]["text"])

    def test_fundamentals_and_risk_become_readable_evidence(self):
        fund = lambda s: {"data": {"history_newest_first": [
            {"fy": "2026-03-31", "revenue_cr": 267021.0, "ebit_margin_pct": 25.0, "roce_pct": 54.9, "fcf_cr": 47948.0}]}}
        risk = lambda s: {"data": {"volatility_pct": 18.0, "beta": 0.83, "pct_vs_200dma": -4.0}}
        r, _ = self.run_loop([act("get_fundamentals_history"), act("get_risk_profile"), act("finish", findings=[])],
                             tools=tools(fundamentals=fund, risk=risk))
        self.assertIn("ROCE 54.9%", r["evidence"][0]["text"])
        self.assertIn("beta against NIFTY 50 0.83", r["evidence"][1]["text"])


class DocumentAvailability(unittest.TestCase):
    """Regression: a live run investigated RELIANCE and TITAN (no documents) and ignored TCS (has a transcript)."""

    def setUp(self):
        ledger.reset()

    def test_without_documents_the_tool_is_not_offered_and_the_prompt_says_so(self):
        llm = Scripted([act("finish", findings=[])])
        investigate_stock(llm, tools(), "RELIANCE", "Reliance", "Energy", "why?", FLAGS, doc_sources=[])
        self.assertIn("No filings or transcripts are available", llm.prompts[0])
        self.assertNotIn("- search_documents", llm.prompts[0])

    def test_with_documents_the_sources_are_named(self):
        llm = Scripted([act("finish", findings=[])])
        investigate_stock(llm, tools(), "TCS", "TCS", "IT", "why?", FLAGS, doc_sources=["call.pdf"])
        self.assertIn("DOCUMENTS AVAILABLE FOR THIS COMPANY: call.pdf", llm.prompts[0])
        self.assertIn("- search_documents", llm.prompts[0])

    def test_a_search_request_without_documents_is_answered_honestly(self):
        llm = Scripted([act("search_documents", args={"query": "x"}), act("finish", findings=[])])
        r = investigate_stock(llm, tools(), "ITC", "ITC", "FMCG", "why?", FLAGS, doc_sources=[])
        self.assertIn("No documents are available", r["evidence"][0]["text"])

    def test_triage_is_told_which_stocks_have_documents(self):
        flags = {"A": evaluate_stock("A", {"roce_pct": 9.0}), "B": evaluate_stock("B", {"roce_pct": 9.0})}
        llm = Scripted([json.dumps({"investigate": [{"symbol": "A", "question": "q"}]})])
        triage(llm, flags, {}, max_stocks=1, has_docs={"B"})
        self.assertIn("B (B), concern score", llm.prompts[0])
        self.assertIn("filings and transcripts available", llm.prompts[0])
        self.assertIn("no filings available", llm.prompts[0])

    def test_a_flagged_stock_with_documents_is_swapped_in_when_the_model_ignores_it(self):
        flags = {"A": evaluate_stock("A", {"roce_pct": 9.0, "pct_vs_200dma": -3.0}),
                 "TCS": evaluate_stock("TCS", {"roce_pct": 9.0})}
        llm = Scripted([json.dumps({"investigate": [{"symbol": "A", "question": "q?"}]})])
        picks, _, how = triage(llm, flags, {}, max_stocks=1, has_docs={"TCS"})
        self.assertEqual([p["symbol"] for p in picks], ["TCS"])
        self.assertIn("added because it is the flagged stock that has documents", how)

    def test_no_swap_when_the_model_already_chose_a_stock_with_documents(self):
        flags = {"A": evaluate_stock("A", {"roce_pct": 9.0}), "TCS": evaluate_stock("TCS", {"roce_pct": 9.0})}
        llm = Scripted([json.dumps({"investigate": [{"symbol": "TCS", "question": "q?"}]})])
        picks, _, how = triage(llm, flags, {}, max_stocks=1, has_docs={"TCS"})
        self.assertEqual((picks[0]["symbol"], how), ("TCS", "model"))


class Findings(unittest.TestCase):
    EVIDENCE = [{"id": "E1", "tool": "search_documents", "text": DOC["text"], "source": "call.pdf", "page": 8}]

    def setUp(self):
        ledger.reset()
        ledger.record("search_documents", {}, {"passages": [{"text": DOC["text"]}]})

    def check(self, claim, evidence):
        return check_findings([{"claim": claim, "evidence": evidence}], self.EVIDENCE, ledger.snapshot())[0]

    def test_supported_claim(self):
        self.assertTrue(self.check("Operating margin was 25%, an expansion of 70 basis points.", ["E1"])["verified"])

    def test_invented_figure(self):
        f = self.check("Operating margin was 31%, an expansion of 70 basis points.", ["E1"])
        self.assertFalse(f["verified"])
        self.assertTrue(any("31" in i for i in f["issues"]))

    def test_citation_to_missing_evidence(self):
        f = self.check("Operating margin was 25%.", ["E9"])
        self.assertTrue(any("does not exist" in i for i in f["issues"]))

    def test_citation_that_does_not_support_the_claim(self):
        f = self.check("Employee attrition stabilised across the delivery centres this quarter.", ["E1"])
        self.assertFalse(f["verified"])
        self.assertTrue(any("does not appear to support" in i for i in f["issues"]))

    def test_uncited_claim_is_flagged_but_an_honest_gap_is_not(self):
        self.assertFalse(self.check("Margins will keep expanding next year.", [])["verified"])
        gap = self.check("The documents do not explain the decline in ROCE.", [])
        self.assertTrue(gap["verified"])
        self.assertEqual(gap["kind"], "gap")

    def test_an_absence_statement_that_cites_its_evidence_is_not_graded_as_a_claim(self):
        """Regression: 'the evidence does not contain X [E1]' was flagged with '18% word overlap'."""
        f = self.check("The provided history shows ROCE fluctuating, but the available evidence does not contain "
                       "data on whether capital expenditure explains it.", ["E1"])
        self.assertTrue(f["verified"], f["issues"])
        self.assertEqual(f["kind"], "gap")

    def test_an_absence_statement_still_has_its_figures_checked(self):
        f = self.check("The evidence does not explain why margin fell to 31%.", ["E1"])
        self.assertFalse(f["verified"])

    def test_at_most_three_findings(self):
        raw = [{"claim": f"Claim {i} operating margin 25%", "evidence": ["E1"]} for i in range(6)]
        self.assertEqual(len(check_findings(raw, self.EVIDENCE, ledger.snapshot())), 3)


if __name__ == "__main__":
    unittest.main()
