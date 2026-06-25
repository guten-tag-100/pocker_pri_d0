"""Self-contained inference for the Poker44 miner.

Loads the trained artifact and turns a list of chunks (each a list of hands) into
one calibrated risk_score per chunk. This is the ONLY module the miner needs to
import. Feature extraction here MUST be byte-identical to training (it imports the
same features.py), otherwise scores drift.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np

# v2 (hero-free, sanitization-invariant) features are what we serve; selected via
# meta.feature_version. The legacy v1 extractor is optional (kept only for old
# artifacts) and may not be present in a cleaned deployment.
from features_v2 import extract_features_v2

try:  # legacy v1 fallback, optional
    from features import extract_features
except Exception:
    extract_features = None

_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _remap_to_threshold(p: np.ndarray, t: float) -> np.ndarray:
    """Monotonic map so the deploy threshold t lands at 0.5 (preserves ranking/AP).

    The validator computes prediction = round(score), so the operating point must
    sit at 0.5. Ranking is unchanged, so AP is unaffected; only the bot/human
    decision boundary moves to where it keeps fpr under the safety cliff."""
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    out = np.where(p >= t, 0.5 + 0.5 * (p - t) / (1 - t), 0.5 * p / t)
    return np.clip(out, 0.0, 1.0)


class Poker44Model:
    def __init__(self, art_dir: str = _ART):
        with open(os.path.join(art_dir, "model.pkl"), "rb") as fh:
            self.model = pickle.load(fh)
        with open(os.path.join(art_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.cols: List[str] = self.meta["feature_order"]
        self.threshold: float = self.meta["deploy_threshold"]
        # pick the feature extractor that matches how the artifact was trained.
        # v2 = hero-free, sanitization-invariant (matches the live validator feed).
        self.feature_version: str = self.meta.get("feature_version", "v1")
        if self.feature_version == "v2":
            self._extract = extract_features_v2
        elif extract_features is not None:
            self._extract = extract_features
        else:
            raise RuntimeError(
                "artifact uses v1 features but features.py is not available")

    def _vectorize(self, chunk: List[Dict[str, Any]]) -> np.ndarray:
        feats = self._extract(chunk or [])
        return np.array([float(feats.get(c, 0.0)) for c in self.cols], dtype=float)

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        """One risk_score in [0,1] per chunk (high = bot-like)."""
        if not chunks:
            return []
        X = np.vstack([self._vectorize(c) for c in chunks])
        p = self.model.predict_proba(X)[:, 1]
        scores = _remap_to_threshold(p, self.threshold)
        # defensive: a degenerate/empty chunk must not be flagged as a bot
        # (protects the human-safety FPR cliff on malformed input)
        out = []
        for chunk, s in zip(chunks, scores):
            if not chunk:
                out.append(0.1)
            else:
                out.append(round(float(s), 6))
        return out


_SINGLETON: Poker44Model | None = None


def get_model() -> Poker44Model:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Poker44Model()
    return _SINGLETON


if __name__ == "__main__":
    # smoke test on real cached data
    from dataset import load_examples

    ex = load_examples()
    m = get_model()
    chunks = [e.hands for e in ex[:12]]
    labels = [e.label for e in ex[:12]]
    scores = m.score_chunks(chunks)
    print(f"threshold={m.threshold:.4f}  cv_ap={m.meta['cv_ap']:.4f}  cv_reward={m.meta['cv_reward']:.4f}")
    print("score | pred | label")
    for s, l in zip(scores, labels):
        print(f"{s:.4f} |  {int(s>=0.5)}   |  {l}")
