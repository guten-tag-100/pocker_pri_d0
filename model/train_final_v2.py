"""Train the v2 production model: hero-free sanitization-invariant features
(features_v2) on the validator's SANITIZED payload, a soft-voting ExtraTrees +
HistGradientBoosting ensemble, with a conservative conformal deploy threshold.

Train == serve: every training chunk is passed through
`poker44.validator.payload_view.prepare_hand_for_miner` before featurizing, so
the model sees the same distribution the validator serves live. This closes the
CV-0.82-but-live-0.42 gap of the v1 hero-based model.

Saves artifacts/model.pkl + meta.json (feature_version="v2", trained_on="sanitized").
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              VotingClassifier)
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, "/root/pocker")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

from dataset import load_examples  # noqa: E402
from evaluate import fpr_target_threshold, remap_to_threshold  # noqa: E402
from features_v2 import extract_features_v2, feature_names_v2  # noqa: E402
from reward_fn import reward  # noqa: E402

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
TARGET_FPR = 0.03


def make_model():
    """Soft-voting ExtraTrees + HistGradientBoosting (both ~0.87 AP, decorrelated)."""
    et = ExtraTreesClassifier(
        n_estimators=600, max_depth=12, min_samples_leaf=2, n_jobs=-1,
        random_state=0, class_weight="balanced_subsample")
    hgb = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.03, max_iter=600, l2_regularization=2.0,
        random_state=0)
    return VotingClassifier([("et", et), ("hgb", hgb)], voting="soft", n_jobs=1)


def sanitize_chunk(hands):
    out = []
    for h in hands:
        try:
            out.append(prepare_hand_for_miner(h))
        except Exception:
            out.append(h)
    return out


def build_matrix(examples, cols=None):
    rows = [extract_features_v2(sanitize_chunk(e.hands)) for e in examples]
    X = pd.DataFrame(rows).fillna(0.0)
    if cols is None:
        cols = sorted(X.columns)
    X = X.reindex(columns=cols, fill_value=0.0)
    return X, cols


if __name__ == "__main__":
    os.makedirs(ART, exist_ok=True)
    t0 = time.time()
    ex = load_examples()
    cols = feature_names_v2(sanitize_chunk(ex[0].hands))
    X, cols = build_matrix(ex, cols)
    y = np.array([e.label for e in ex])
    dates = np.array([e.source_date for e in ex])
    print(f"train(sanitized): {X.shape[0]} chunks, {X.shape[1]} v2 features, "
          f"{len(set(dates))} dates  ({time.time()-t0:.0f}s)", flush=True)

    # one cross-date OOF pass = honest metrics AND the conservative deploy threshold
    oof = cross_val_predict(make_model(), X.values, y, groups=dates,
                            cv=GroupKFold(6), method="predict_proba")[:, 1]
    cv_ap = float(average_precision_score(y, oof))
    cv_auc = float(roc_auc_score(y, oof))
    deploy_t = fpr_target_threshold(oof[y == 0], TARGET_FPR)
    rew, res = reward(remap_to_threshold(oof, deploy_t), y)
    out = {"ap": cv_ap, "auc": cv_auc, "reward": float(rew),
           "fpr": float(res["fpr"]), "recall": float(res["bot_recall"])}
    print(f"OOF: cv_ap={cv_ap:.4f} reward={rew:.4f} fpr={res['fpr']:.4f} "
          f"recall={res['bot_recall']:.4f} thr={deploy_t:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    model = make_model().fit(X.values, y)
    with open(os.path.join(ART, "model.pkl"), "wb") as fh:
        pickle.dump(model, fh)
    meta = {
        "feature_version": "v2",
        "trained_on": "sanitized",
        "feature_order": cols,
        "deploy_threshold": float(deploy_t),
        "target_fpr": TARGET_FPR,
        "model": "VotingClassifier(soft: ExtraTrees600d12 + HistGBM d4 lr.03 it600)",
        "cv_ap": out["ap"],
        "cv_auc": out["auc"],
        "cv_reward": out["reward"],
        "cv_fpr": out["fpr"],
        "cv_recall": out["recall"],
        "n_train": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_dates": int(len(set(dates))),
        "benchmark_releases": sorted(set(dates.tolist())),
    }
    with open(os.path.join(ART, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nsaved artifacts/model.pkl ({os.path.getsize(os.path.join(ART,'model.pkl'))//1024} KB)")
    print(f"saved artifacts/meta.json | feature_version=v2 cv_ap={meta['cv_ap']:.4f} "
          f"cv_reward={meta['cv_reward']:.4f} cv_fpr={meta['cv_fpr']:.4f}")
