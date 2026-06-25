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

# Per-batch positive budget: cap how many chunks may be predicted as bots in one
# validator batch. The reward is (0.65*AP + 0.35*recall) * (1-fpr)^2, HARD-ZEROED
# at fpr>=0.10. AP (ranking) dominates and is threshold-independent, so the top
# miners predict FEW positives (recall ~0-0.33) to keep fpr near 0 (safety=1) and
# win on AP. Without this cap our well-ranked model flags ~all chunks (bots=38/40)
# and trips the human-safety cliff -> reward 0. This monotonically demotes all but
# the top-K (by score, and only if above the deploy boundary) below 0.5, which
# PRESERVES the ranking (AP unchanged) while bounding the false-positive rate.
_MAX_POS_FRAC = float(os.environ.get("POKER44_MAX_POS_FRAC", "0.15"))


def _apply_batch_safety_budget(scores: np.ndarray, max_frac: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0 or max_frac >= 1.0:
        return s
    k = int(np.floor(max_frac * n))
    order = np.argsort(-s, kind="stable")          # indices, highest score first
    allowed = {int(i) for i in order[:k] if s[i] >= 0.5}   # top-K that are positive
    out = s.copy()
    squeeze = [int(i) for i in order if int(i) not in allowed]  # everything else
    m = len(squeeze)
    for rank, idx in enumerate(squeeze):           # map into [0,0.499], order kept
        out[idx] = 0.499 * (1.0 - rank / max(m - 1, 1))
    return np.clip(out, 0.0, 1.0)


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
        # cap positives per batch so we never trip the fpr>=0.10 reward cliff
        # (ranking/AP preserved; only the bot/human boundary tightens)
        scores = _apply_batch_safety_budget(scores, _MAX_POS_FRAC)
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
