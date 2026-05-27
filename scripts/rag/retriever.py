"""Phase 2 of the RAG layer: load the index and answer retrieval queries.

The Retriever loads three files built by build_index.py:
    data/rag/docs.jsonl           the documents (text + metadata)
    data/rag/embeddings.npy       BioBERT vectors, row-aligned to docs
    data/rag/gene_to_doc_ids.json gene_id → [doc_id, ...] reverse index

Then exposes one public method:

    retriever.retrieve(query, k=8, gene_filter=None) -> list[dict]

    query       : free-text question (embedded with BioBERT at query time)
    k           : number of top documents to return
    gene_filter : optional set/list of gene IDs (ENSG or Entrez); when
                  provided, only documents that mention at least one of
                  those genes are candidates.  Shrinks the search space
                  dramatically for gene-centric queries.

Each returned document is the original dict from docs.jsonl plus two
extra keys:
    score  : cosine similarity to the query  (float, 0–1)
    rank   : 1-based rank in this result set (int)

DESIGN:
    Vector store: plain numpy cosine — corpus is ~3 k docs, no need for
    FAISS.  At 3k × 768 float32 the full matrix is ~9 MB in RAM.

    Embedder: the same BioBERT model used by build_index.py.  Using the
    same model guarantees query and document vectors live in the same
    space.  The tokenizer + model are lazy-loaded on first retrieve() call
    and cached on the instance.

    Gene filter: applied BEFORE cosine scoring so we never embed the query
    against documents the caller doesn't want.  This makes gene-centric
    retrieval fast even if the corpus grows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# default paths (relative to project root)
# ---------------------------------------------------------------------------

DEFAULT_INDEX_DIR = Path("data/rag")
DEFAULT_BIOBERT = "dmis-lab/biobert-v1.1"
DEFAULT_MAX_LEN = 256


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """Loads the RAG index once and answers cosine-similarity queries."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        biobert_model: str = DEFAULT_BIOBERT,
        max_seq_len: int = DEFAULT_MAX_LEN,
        device: str = "cpu",
    ) -> None:
        self.biobert_model = biobert_model
        self.max_seq_len = max_seq_len
        self.device = device

        # Load documents
        docs_path = index_dir / "docs.jsonl"
        if not docs_path.exists():
            raise FileNotFoundError(
                f"RAG index not found at {docs_path}. "
                "Run `python -m scripts.rag.build_index` first."
            )
        self.docs: list[dict] = [
            json.loads(line)
            for line in docs_path.read_text().splitlines()
            if line.strip()
        ]

        # Load embeddings — shape (N, D), float32
        emb_path = index_dir / "embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {emb_path}. "
                "Run `python -m scripts.rag.build_index` without --skip-embed."
            )
        self._embeddings: np.ndarray = np.load(emb_path)  # (N, D)

        # Pre-normalise rows so cosine = dot product
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
        self._embeddings_norm = self._embeddings / norms

        # Load gene reverse index
        rev_path = index_dir / "gene_to_doc_ids.json"
        self._gene_to_doc_ids: dict[str, list[str]] = (
            json.loads(rev_path.read_text()) if rev_path.exists() else {}
        )

        # Build doc_id → row-index map for fast look-ups
        self._id_to_row: dict[str, int] = {d["id"]: i for i, d in enumerate(self.docs)}

        # BioBERT tokenizer + model — lazy loaded on first retrieve()
        self._tokenizer: Optional[object] = None
        self._model: Optional[object] = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 8,
        gene_filter: Optional[list[str] | set[str]] = None,
    ) -> list[dict]:
        """Return top-k documents most similar to query.

        Parameters
        ----------
        query
            Free-text question or phrase to embed and search against.
        k
            Number of documents to return.
        gene_filter
            If provided, only consider documents that contain at least one
            of these gene IDs (ENSG or Entrez string IDs).
        """
        # 1. Determine candidate row indices
        if gene_filter:
            candidate_ids: set[str] = set()
            for gid in gene_filter:
                candidate_ids.update(self._gene_to_doc_ids.get(str(gid), []))
            if not candidate_ids:
                # No docs cover these genes — fall back to full corpus
                candidate_rows = np.arange(len(self.docs))
            else:
                candidate_rows = np.array(
                    [
                        self._id_to_row[doc_id]
                        for doc_id in candidate_ids
                        if doc_id in self._id_to_row
                    ],
                    dtype=np.int64,
                )
        else:
            candidate_rows = np.arange(len(self.docs))

        # 2. Embed the query
        q_vec = self._embed_query(query)  # shape (D,)

        # 3. Cosine similarity against candidate rows
        candidate_embs = self._embeddings_norm[candidate_rows]  # (M, D)
        scores = candidate_embs @ q_vec  # (M,)

        # 4. Top-k
        top_k = min(k, len(scores))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]  # fast partial sort
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]  # sort descending

        # 5. Assemble results
        results = []
        for rank, local_idx in enumerate(top_idx, start=1):
            row = int(candidate_rows[local_idx])
            score = float(scores[local_idx])
            doc = dict(self.docs[row])  # shallow copy so we don't mutate
            doc["score"] = round(score, 4)
            doc["rank"] = rank
            results.append(doc)

        return results

    def retrieve_for_genes(
        self,
        genes: list[str],
        k: int = 8,
        query: str = "",
    ) -> list[dict]:
        """Convenience wrapper: retrieve docs for a gene list.

        If `query` is empty, synthesises a query from the gene names.
        Useful when you have a DEG list but no specific question.
        """
        if not query:
            query = (
                f"biological pathways and functions of genes: {', '.join(genes[:20])}"
            )
        return self.retrieve(query=query, k=k, gene_filter=genes)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a single text with BioBERT, mean-pooled. Returns (D,)."""
        self._lazy_load_biobert()
        import torch

        assert self._tokenizer is not None and self._model is not None
        enc = self._tokenizer(  # type: ignore[operator]
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            hidden = self._model(**enc).last_hidden_state  # type: ignore[operator]
            mask = enc["attention_mask"].unsqueeze(-1)  # (1, L, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        vec = pooled[0].cpu().numpy()  # (D,)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _lazy_load_biobert(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise SystemExit(
                f"transformers not installed: {e}\n" "pip install transformers torch"
            ) from e
        self._tokenizer = AutoTokenizer.from_pretrained(self.biobert_model)
        self._model = (
            __import__("transformers")
            .AutoModel.from_pretrained(self.biobert_model)
            .to(self.device)
            .eval()
        )


# ---------------------------------------------------------------------------
# module-level singleton (lazy) — shared by any code that imports retriever
# ---------------------------------------------------------------------------

_default_retriever: Optional[Retriever] = None


def get_retriever(
    index_dir: Path = DEFAULT_INDEX_DIR,
    biobert_model: str = DEFAULT_BIOBERT,
) -> Retriever:
    """Return a cached Retriever instance (loads index once per process)."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever(index_dir=index_dir, biobert_model=biobert_model)
    return _default_retriever


# ---------------------------------------------------------------------------
# quick CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "cell cycle regulation CDK4 CCND1"
    print(f"Query: {query!r}\n")
    r = Retriever()
    hits = r.retrieve(query, k=5)
    for h in hits:
        print(
            f"[{h['rank']}] score={h['score']:.4f}  "
            f"{h['source']:10s}  "
            f"{h['pathway_name']}"
        )
        print(f"     {h['text'][:120]}...")
        print()
