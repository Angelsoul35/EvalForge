"""Lexical similarity used by the deterministic judge.

It is a TF-IDF cosine similarity over word n-grams. It is deterministic and
needs no model download and no network, so the whole tool runs and is tested
offline. This is lexical, not semantic: you could later plug in real embeddings
behind the same functions, but the default is explicit about what it does.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences/claims (coarse but deterministic)."""
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class TfidfSimilarity:
    """Fit a vectorizer on a corpus, then score cosine similarity of texts."""

    def __init__(self, corpus: list[str]) -> None:
        cleaned = [c for c in corpus if c and c.strip()]
        self._vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, stop_words=None
        )
        if cleaned:
            self._vectorizer.fit(cleaned)
            self._fitted = True
        else:
            self._fitted = False

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity in [0, 1] between two texts."""
        if not self._fitted or not a.strip() or not b.strip():
            return 0.0
        vecs = self._vectorizer.transform([a, b])
        score = float(cosine_similarity(vecs[0], vecs[1])[0, 0])
        return max(0.0, min(1.0, score))

    def max_similarity(self, text: str, candidates: list[str]) -> float:
        """Highest similarity of ``text`` against any candidate."""
        if not candidates:
            return 0.0
        return max(self.similarity(text, c) for c in candidates)


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 (SQuAD-style), a simple correctness proxy."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common: dict[str, int] = {}
    pred_counts = _counts(pred_tokens)
    ref_counts = _counts(ref_tokens)
    for tok, c in pred_counts.items():
        if tok in ref_counts:
            common[tok] = min(c, ref_counts[tok])
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return float(2 * precision * recall / (precision + recall))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _counts(tokens: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tokens:
        out[t] = out.get(t, 0) + 1
    return out


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
