"""BM25 sparse index tren table card.

Tu implement BM25Okapi (~40 dong) thay vi keo them rank_bm25 — de kiem soat
tokenizer tieng Viet va tranh mot dependency nua.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..utils.io import load_pickle, save_pickle
from ..utils.logging import get_logger
from ..utils.spell_check import tokenize

log = get_logger(__name__)


class BM25Store:
    """BM25 Okapi. Score bang 0 khi khong khop token nao."""

    def __init__(self, k1: float | None = None, b: float | None = None):
        cfg = get_settings().retrieval.get("bm25", {})
        self.k1 = k1 if k1 is not None else cfg.get("k1", 1.5)
        self.b = b if b is not None else cfg.get("b", 0.75)

        self.ids: list[str] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._lengths: np.ndarray = np.zeros(0)
        self._avg_len: float = 0.0

    # ── build ─────────────────────────────────────────────

    def build(self, ids: list[str], texts: list[str]) -> "BM25Store":
        if len(ids) != len(texts):
            raise ValueError("ids va texts phai cung do dai")

        self.ids = list(ids)
        self._tf = []
        self._df = Counter()
        lengths: list[int] = []

        for text in texts:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self._tf.append(tf)
            lengths.append(len(tokens))
            self._df.update(tf.keys())

        self._lengths = np.asarray(lengths, dtype=np.float32)
        self._avg_len = float(self._lengths.mean()) if len(lengths) else 0.0

        n = len(self.ids)
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in self._df.items()
        }

        log.info("BM25: %d doc, %d term, avg_len=%.1f", n, len(self._idf), self._avg_len)
        return self

    # ── search ────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 100,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens or not self.ids:
            return []

        scores = np.zeros(len(self.ids), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self._lengths / max(self._avg_len, 1e-9))

        for term in set(tokens):
            idf = self._idf.get(term)
            if idf is None:
                continue
            freqs = np.array([tf.get(term, 0) for tf in self._tf], dtype=np.float32)
            scores += idf * (freqs * (self.k1 + 1)) / (freqs + norm)

        if allowed_ids is not None:
            mask = np.array([i in allowed_ids for i in self.ids])
            scores = np.where(mask, scores, -np.inf)

        k = min(top_k, len(self.ids))
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]

        return [
            (self.ids[i], float(scores[i]))
            for i in order
            if np.isfinite(scores[i]) and scores[i] > 0
        ]

    # ── persist ───────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else get_settings().paths.index / "bm25.pkl"
        save_pickle({
            "k1": self.k1, "b": self.b, "ids": self.ids, "tf": self._tf,
            "df": self._df, "idf": self._idf,
            "lengths": self._lengths, "avg_len": self._avg_len,
        }, target)
        log.info("Da luu BM25 -> %s", target)
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BM25Store":
        target = Path(path) if path else get_settings().paths.index / "bm25.pkl"
        if not target.exists():
            raise FileNotFoundError(f"Chua co BM25 index: {target}")

        state = load_pickle(target)
        store = cls(k1=state["k1"], b=state["b"])
        store.ids = state["ids"]
        store._tf = state["tf"]
        store._df = state["df"]
        store._idf = state["idf"]
        store._lengths = state["lengths"]
        store._avg_len = state["avg_len"]
        return store
