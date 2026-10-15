"""Offline tests for document search, grounded answers and the evaluation scoring.

Uses LlamaIndex with the deterministic hashing embedding, so no model download and no network.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("NSE_AUTO_REFRESH", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.qa import NOT_FOUND, answer_question, citation_report, passages_prompt
from pipeline.rag import DocumentIndex, HashingEmbedding, load_documents
from pipeline.rag_eval import (ANSWERABLE, UNANSWERABLE, batch_prompt, parse_answers, score_abstention,
                               score_answer)

TCS_TEXT = ("For FY26 our operating margin was 25 percent an expansion of 70 basis points over the prior year and at a "
            "four year high. We delivered 40.7 billion dollars in total contract value including 5 mega deals. "
            "The board recommended a final dividend of 31 rupees per share. ") * 2
INFY_TEXT = ("Infosys reported large deal wins and the operating margin guidance for the year is between 20 and 22 "
             "percent. Attrition has stabilised across the delivery centres this quarter and the order pipeline is healthy. ") * 2
GENERAL = "Our investment policy says no single sector should exceed forty percent of the portfolio at any time. " * 3


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.reply, {"provider": "fake", "model": "scripted"}


class Search(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        root = Path(cls.dir.name)
        (root / "tcs").mkdir()
        (root / "infy").mkdir()
        (root / "tcs" / "call.txt").write_text(TCS_TEXT)
        (root / "infy" / "call.txt").write_text(INFY_TEXT)
        (root / "policy.md").write_text(GENERAL)
        (root / "README.md").write_text("Instructions for humans, not research.")
        cls.index = DocumentIndex.build(root, embed_model=HashingEmbedding(), backend="hashing", persist=False)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_folder_name_becomes_the_company(self):
        docs = load_documents(self.dir.name)
        self.assertEqual({d.metadata["symbol"] for d in docs}, {"TCS", "INFY", ""})
        self.assertNotIn("README.md", {d.metadata["source"] for d in docs})

    def test_search_is_scoped_to_the_company(self):
        hits = self.index.search("operating margin", k=5, symbol="TCS")
        self.assertTrue(hits)
        self.assertTrue(all(h["symbol"] in ("TCS", None) for h in hits))
        self.assertNotIn("Infosys", " ".join(h["text"] for h in hits))

    def test_general_documents_apply_to_every_company(self):
        hits = self.index.search("no single sector should exceed forty percent", k=3, symbol="TCS")
        self.assertIn("policy.md", {h["source"] for h in hits})

    def test_hits_carry_source_page_and_how_they_were_found(self):
        h = self.index.search("total contract value mega deals", k=1, symbol="TCS")[0]
        self.assertEqual(h["source"], "call.txt")
        self.assertIn("40.7", h["text"])
        self.assertIn(h["found_by"], ("keywords", "meaning", "keywords+meaning"))

    def test_keywords_rescue_an_exact_term(self):
        hybrid = self.index.search("mega deals", k=1, symbol="TCS", hybrid=True)
        self.assertIn("mega deals", hybrid[0]["text"])

    def test_empty_knowledge_folder(self):
        with tempfile.TemporaryDirectory() as d:
            idx = DocumentIndex.build(d, embed_model=HashingEmbedding(), backend="hashing", persist=False)
            self.assertEqual(idx.search("anything"), [])
            self.assertEqual(len(idx), 0)

    def test_index_is_cached_and_rebuilt_when_files_change(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.txt").write_text(TCS_TEXT)
            first = DocumentIndex.build(d, embed_model=HashingEmbedding(), backend="hashing")
            self.assertTrue((Path(d) / ".index" / "manifest.json").exists())
            again = DocumentIndex.build(d, embed_model=HashingEmbedding(), backend="hashing")
            self.assertEqual(first.n_chunks, again.n_chunks)
            Path(d, "b.txt").write_text(INFY_TEXT)
            changed = DocumentIndex.build(d, embed_model=HashingEmbedding(), backend="hashing")
            self.assertGreater(changed.n_chunks, first.n_chunks)


class GroundedAnswers(unittest.TestCase):
    PASSAGES = [{"source": "call.txt", "page": 8,
                 "text": "For FY26 our operating margin was 25 percent, an expansion of 70 basis points, a four year high."},
                {"source": "call.txt", "page": 5,
                 "text": "We delivered 40.7 billion dollars in total contract value including 5 mega deals."}]

    class Index:
        def __init__(self, passages):
            self.passages = passages

        def search(self, q, k=3, symbol=None):
            return self.passages[:k]

    def ask(self, reply, passages=None):
        idx = self.Index(self.PASSAGES if passages is None else passages)
        return answer_question("What was the margin?", idx, FakeLLM(reply), symbol="TCS")

    def test_a_cited_and_supported_answer_is_grounded(self):
        r = self.ask("The operating margin was 25 percent, up 70 basis points [D1].")
        self.assertTrue(r["grounded"])
        self.assertFalse(r["abstained"])

    def test_a_wrong_figure_is_caught(self):
        r = self.ask("The operating margin was 31 percent [D1].")
        self.assertFalse(r["grounded"])
        self.assertGreater(r["verification"]["n_problems"], 0)

    def test_a_citation_to_a_passage_that_does_not_exist_is_caught(self):
        r = self.ask("The operating margin was 25 percent [D7].")
        self.assertFalse(r["grounded"])
        self.assertEqual(r["citations"][0]["missing"], [7])

    def test_a_citation_that_does_not_support_the_claim_is_caught(self):
        r = self.ask("Employee attrition stabilised across delivery centres [D2].")
        self.assertFalse(r["citations"][0]["ok"])
        self.assertFalse(r["grounded"])

    def test_an_uncited_claim_is_not_grounded(self):
        r = self.ask("The operating margin was 25 percent.")
        self.assertTrue(r["uncited"])
        self.assertFalse(r["grounded"])

    def test_abstention_is_recognised_and_needs_no_citation(self):
        r = self.ask(NOT_FOUND)
        self.assertTrue(r["abstained"])
        self.assertEqual(r["citations"], [])

    def test_no_passages_means_no_model_call(self):
        llm = FakeLLM("should not be used")
        r = answer_question("q", self.Index([]), llm)
        self.assertTrue(r["abstained"])
        self.assertEqual(llm.prompts, [])

    def test_prompt_numbers_the_passages_and_forbids_outside_knowledge(self):
        p = passages_prompt("q?", self.PASSAGES)
        self.assertIn("[D1] (call.txt, page 8)", p)
        self.assertIn("[D2]", p)
        self.assertIn(NOT_FOUND, p)

    def test_citation_report_support_ratio(self):
        rep = citation_report("Margin was 25 percent [D1]. Totally unrelated words appear here [D2].", self.PASSAGES)
        self.assertTrue(rep[0]["ok"])
        self.assertFalse(rep[1]["ok"])


class EvalScoring(unittest.TestCase):
    def test_answerable_needs_every_pattern(self):
        _, pats, _ = ANSWERABLE[0]                      # contract value + mega deals
        self.assertTrue(score_answer("TCV of $40.7B including 5 mega deals", pats))
        self.assertFalse(score_answer("TCV of $40.7B", pats))
        self.assertFalse(score_answer(NOT_FOUND, pats))

    def test_abstention_scoring(self):
        self.assertTrue(score_abstention(NOT_FOUND))
        self.assertTrue(score_abstention("This is not mentioned in the call."))
        self.assertFalse(score_abstention("Attrition was 12.5%."))

    def test_batch_prompt_and_parsing_roundtrip(self):
        qp = [("What is X?", [{"page": 2, "text": "X is 5."}])]
        self.assertIn("QUESTION 1: What is X?", batch_prompt(questions_with_passages=qp))
        self.assertIn("from your own knowledge", batch_prompt(questions=["What is X?"]))
        reply = '```json\n{"answers": [{"n": 1, "answer": "A"}, {"n": 3, "answer": "C"}]}\n```'
        self.assertEqual(parse_answers(reply, 3), ["A", "", "C"])
        self.assertEqual(parse_answers("garbage", 2), ["", ""])

    def test_question_sets_are_sane(self):
        self.assertEqual(len(ANSWERABLE), 10)
        self.assertEqual(len(UNANSWERABLE), 4)
        self.assertEqual(len({q for q, _, _ in ANSWERABLE} | set(UNANSWERABLE)), 14)


if __name__ == "__main__":
    unittest.main()
