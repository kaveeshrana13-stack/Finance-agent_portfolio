"""A small document search (BM25) over files in knowledge/ — annual reports, concall notes, policy notes.

Supports .txt, .md and text-based .pdf. Every passage keeps its source file and page so an answer can
cite it. BM25 is keyword-based (no embeddings), which keeps it free, offline and explainable.
"""
import math
import re
from collections import Counter
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9&]+")


def _tokens(text):
    return _TOKEN.findall(text.lower())


def chunk(text, size=110, overlap=25):
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


class _BM25:
    """Okapi BM25 with the smoothed idf Lucene uses, log(1 + (N - n + 0.5) / (n + 0.5)).

    The textbook idf is zero (or negative) for a term found in half the documents, which made
    every query on a small knowledge base score 0.
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(self.len)) if docs else 0
        self.k1, self.b, self.n = k1, b, len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens):
        out = []
        for doc, length in zip(self.docs, self.len):
            s = 0.0
            for t in query_tokens:
                f = doc.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
            out.append(s)
        return out


class KnowledgeBase:
    def __init__(self, passages=None):
        self.passages = list(passages or [])        # [{source, page, text}]
        self._index = _BM25([_tokens(p["text"]) for p in self.passages]) if self.passages else None

    @classmethod
    def from_directory(cls, root):
        passages = []
        for path in sorted(Path(root).glob("**/*")):
            if path.name.lower().startswith("readme"):
                continue                                   # instructions for humans, not research
            if path.suffix.lower() in (".txt", ".md"):
                pages = [(None, path.read_text(encoding="utf-8", errors="ignore"))]
            elif path.suffix.lower() == ".pdf":
                try:
                    from pypdf import PdfReader
                    pages = [(i + 1, (pg.extract_text() or "")) for i, pg in enumerate(PdfReader(str(path)).pages)]
                except Exception:
                    continue
            else:
                continue
            for page, text in pages:
                for piece in chunk(text):
                    passages.append({"source": path.name, "page": page, "text": piece})
        return cls(passages)

    def __len__(self):
        return len(self.passages)

    def search(self, query, k=3, min_score=0.3):
        if not self._index:
            return []
        scores = self._index.scores(_tokens(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.passages[i], "score": round(float(scores[i]), 2)} for i in ranked if scores[i] >= min_score]
