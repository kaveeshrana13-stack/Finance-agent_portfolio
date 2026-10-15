"""Meaning-based search over company documents, built on LlamaIndex.

    knowledge/
        tcs/                         <- folder name = the NSE symbol the documents belong to
            earnings_call_q4_fy26.pdf
        general_policy_note.md       <- documents with no folder apply to every company

LlamaIndex does what the lessons covered: Documents -> SentenceSplitter nodes -> embeddings ->
VectorStoreIndex -> retriever. Embeddings come from a small local model (BAAI/bge-small-en-v1.5 through
FastEmbed, free, no API quota). Answer generation is NOT done by LlamaIndex: the project's own model chain
writes the answer so that it can be verified against the retrieved passages and use the free fallbacks.

Each hit carries its source file and page so a claim can be cited as [D1] and checked.
"""
import hashlib
import json
import math
import re
from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter

from pipeline.knowledge import _BM25, _tokens

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS, CHUNK_OVERLAP = 220, 40
_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedding(BaseEmbedding):
    """Deterministic bag-of-words hashing embedding. No download, no network.

    Used by tests and as a last-resort fallback. It matches shared words, not meaning, so it is
    equivalent to keyword search and is reported as such via DocumentIndex.backend.
    """
    dim: int = 384

    def _vec(self, text):
        v = [0.0] * self.dim
        for t in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def _get_query_embedding(self, query):
        return self._vec(query)

    def _get_text_embedding(self, text):
        return self._vec(text)

    async def _aget_query_embedding(self, query):
        return self._vec(query)


def default_embedding():
    """The local FastEmbed model; falls back to hashing (keyword-like) if it cannot be loaded."""
    try:
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        return FastEmbedEmbedding(model_name=EMBED_MODEL), f"fastembed:{EMBED_MODEL}"
    except Exception as e:                                       # pragma: no cover - environment dependent
        return HashingEmbedding(), f"hashing (fastembed unavailable: {type(e).__name__})"


def _read_file(path):
    """[(page_or_None, text)] for a .pdf / .txt / .md file."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    if suffix == ".pdf":
        from pypdf import PdfReader
        return [(i + 1, (pg.extract_text() or "")) for i, pg in enumerate(PdfReader(str(path)).pages)]
    return []


def load_documents(root):
    """One LlamaIndex Document per page, with metadata for citation and company filtering."""
    root, docs = Path(root), []
    for path in sorted(root.glob("**/*")):
        if not path.is_file() or path.name.lower().startswith("readme") or ".index" in path.parts:
            continue
        rel = path.relative_to(root)
        symbol = rel.parts[0].upper() if len(rel.parts) > 1 else None
        try:
            pages = _read_file(path)
        except Exception:
            continue
        for page, text in pages:
            text = re.sub(r"\s+", " ", text).strip()
            if len(text.split()) >= 15:
                docs.append(Document(text=text, metadata={"source": path.name, "page": page, "symbol": symbol or ""},
                                     excluded_embed_metadata_keys=["source", "page", "symbol"],
                                     excluded_llm_metadata_keys=["source", "page", "symbol"]))
    return docs


def _fingerprint(root, backend):
    items = []
    for p in sorted(Path(root).glob("**/*")):
        if p.is_file() and ".index" not in p.parts and not p.name.lower().startswith("readme"):
            st = p.stat()
            items.append([str(p.relative_to(root)), st.st_size, int(st.st_mtime)])
    return {"files": items, "backend": backend, "chunk": [CHUNK_TOKENS, CHUNK_OVERLAP]}


class DocumentIndex:
    """Search the knowledge folder by meaning. search() returns the same shape as the keyword
    KnowledgeBase, so the rest of the pipeline does not care which one it has."""

    def __init__(self, index, backend, n_docs, n_chunks):
        self._index = index
        self.backend, self.n_docs, self.n_chunks = backend, n_docs, n_chunks
        self._nodes = list(index.docstore.docs.values()) if n_chunks else []
        self._keyword = _BM25([_tokens(n.get_content()) for n in self._nodes]) if self._nodes else None

    @classmethod
    def build(cls, root, embed_model=None, backend=None, persist=True):
        root = Path(root)
        if embed_model is None:
            embed_model, backend = default_embedding()
        backend = backend or type(embed_model).__name__
        persist_dir = root / ".index"
        fp = _fingerprint(root, backend)
        manifest = persist_dir / "manifest.json"

        if persist and manifest.exists():
            try:
                if json.loads(manifest.read_text()) == {**fp}:
                    ctx = StorageContext.from_defaults(persist_dir=str(persist_dir))
                    idx = load_index_from_storage(ctx, embed_model=embed_model)
                    meta = json.loads((persist_dir / "counts.json").read_text())
                    return cls(idx, backend, meta["docs"], meta["chunks"])
            except Exception:
                pass                                              # stale or unreadable cache: rebuild

        docs = load_documents(root)
        splitter = SentenceSplitter(chunk_size=CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(docs)
        idx = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=False)
        if persist and nodes:
            persist_dir.mkdir(parents=True, exist_ok=True)
            idx.storage_context.persist(persist_dir=str(persist_dir))
            (persist_dir / "counts.json").write_text(json.dumps({"docs": len(docs), "chunks": len(nodes)}))
            manifest.write_text(json.dumps(fp))
        return cls(idx, backend, len(docs), len(nodes))

    def __len__(self):
        return self.n_chunks

    def sources(self, symbol=None):
        """File names of the documents that apply to a company (its own folder plus general ones)."""
        out = set()
        for n in self._nodes:
            m = n.metadata or {}
            if not symbol or m.get("symbol") in ("", None, symbol.upper()):
                out.add(m.get("source"))
        return sorted(x for x in out if x)

    def covers(self, symbol):
        """True when company-specific documents exist (general documents alone do not count)."""
        return any((n.metadata or {}).get("symbol") == symbol.upper() for n in self._nodes)

    def search(self, query, k=3, symbol=None, hybrid=True):
        """Top passages: [{source, page, symbol, text, score, found_by}].

        Hybrid retrieval: meaning-based (embedding) results and keyword (BM25) results are merged by
        reciprocal rank fusion. Meaning search alone missed a passage that literally said "TCV" and
        "mega deals" for the query "total contract value and mega deals"; keywords catch exact terms.
        `symbol` keeps that company's documents plus general (folder-less) ones.
        """
        if not self.n_chunks:
            return []
        pool = max(k * 6, 12)
        ranked = {}                                   # node_id -> {"node", "rrf", "by"}

        def take(node, rank, source):
            m = node.metadata or {}
            if symbol and m.get("symbol") not in ("", None, symbol.upper()):
                return
            e = ranked.setdefault(node.node_id, {"node": node, "rrf": 0.0, "by": set()})
            e["rrf"] += 1.0 / (60 + rank)
            e["by"].add(source)

        for r, h in enumerate(self._index.as_retriever(similarity_top_k=pool).retrieve(query)):
            take(h.node, r, "meaning")
        if hybrid and self._keyword:
            scores = self._keyword.scores(_tokens(query))
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool]
            for r, i in enumerate(o for o in order if scores[o] > 0):
                take(self._nodes[i], r, "keywords")

        out = []
        for e in sorted(ranked.values(), key=lambda e: e["rrf"], reverse=True)[:k]:
            m = e["node"].metadata or {}
            out.append({"source": m.get("source"), "page": m.get("page"), "symbol": m.get("symbol") or None,
                        "text": e["node"].get_content(), "score": round(e["rrf"], 4),
                        "found_by": "+".join(sorted(e["by"]))})
        return out
