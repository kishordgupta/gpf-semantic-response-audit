"""Fit the lexical baseline using the original, checksum-locked vocabulary.

NumPy sorting can choose different tied-frequency terms at a max_features cutoff
on different platforms. The released vocabulary preserves the original baseline;
IDF weights are still recomputed from the exact canonical cleaned corpus.
"""
import hashlib
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent


def fit_locked_tfidf(texts):
    texts = list(texts)
    lock = json.loads((ROOT / "results/tfidf_vocabulary.json").read_text())
    terms = lock["terms_by_feature_index"]
    corpus_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()
    vocabulary_hash = hashlib.sha256(json.dumps(terms, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if corpus_hash != lock["text_sha256"]:
        raise ValueError("Cleaned corpus differs from the frozen TF-IDF input; do not reuse this vocabulary for a different dataset.")
    if vocabulary_hash != lock["ordered_vocabulary_sha256"] or len(terms) != lock["features"] or len(set(terms)) != len(terms):
        raise ValueError("TF-IDF vocabulary checksum or feature indexing is inconsistent.")
    parameters = dict(lock["parameters"])
    parameters["ngram_range"] = tuple(parameters["ngram_range"])
    vectorizer = TfidfVectorizer(vocabulary={term: index for index, term in enumerate(terms)}, **parameters)
    return vectorizer.fit_transform(texts)
