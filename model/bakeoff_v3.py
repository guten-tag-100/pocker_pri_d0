"""Approach B bake-off: does a stacked ensemble (LightGBM+XGBoost+CatBoost+ET+RF
-> LogisticRegression meta) beat the current ET+HGB soft-vote, on the SAME
hero-free sanitized features_v2 + honest cross-date CV?

Writes a leaderboard to bakeoff_v3_results.txt. Run in background.
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, "/root/my_pocker/pocker_d0"); sys.path.insert(0, "/root/pocker/model")
from poker44.validator.payload_view import prepare_hand_for_miner
from dataset import load_examples
from features_v2 import extract_features_v2
from evaluate import fpr_target_threshold, remap_to_threshold
from reward_fn import reward
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import average_precision_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, VotingClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb, xgboost as xgb, catboost as cb

HERE = "/root/pocker/model"
OUT = os.path.join(HERE, "bakeoff_v3_results.txt")
CACHE = os.path.join(HERE, "_san_cache_v3.npz")
TARGET_FPR = 0.03


def log(m):
    open(OUT, "a").write(m + "\n"); print(m, flush=True)


def build_cache():
    ex = load_examples()
    y = np.array([e.label for e in ex]); dates = np.array([e.source_date for e in ex])
    t0 = time.time()
    san = [[prepare_hand_for_miner(h) for h in e.hands] for e in ex]
    X = pd.DataFrame([extract_features_v2(c) for c in san]).fillna(0.0)
    X = X.reindex(sorted(X.columns), axis=1)
    np.savez(CACHE, X=X.values, y=y, dates=dates.astype(str), cols=np.array(list(X.columns)))
    log(f"# built sanitized features_v2 cache in {time.time()-t0:.0f}s  X={X.shape}")
    return X.values, y, dates


def load_cache():
    d = np.load(CACHE, allow_pickle=True)
    return d["X"], d["y"], d["dates"]


def lgbm(): return lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.02, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
    subsample_freq=1, n_jobs=-1, random_state=0, verbose=-1)
def xgbc(): return xgb.XGBClassifier(n_estimators=1000, learning_rate=0.025, max_depth=7,
    min_child_weight=5, subsample=0.85, colsample_bytree=0.7, reg_lambda=1.0,
    tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=0)
def catb(): return cb.CatBoostClassifier(iterations=1200, learning_rate=0.03, depth=7,
    l2_leaf_reg=3.0, eval_metric="PRAUC", random_seed=0, verbose=0)
def et(): return ExtraTreesClassifier(n_estimators=900, max_depth=12, min_samples_leaf=1,
    n_jobs=-1, random_state=0, class_weight="balanced_subsample")
def rf(): return RandomForestClassifier(n_estimators=700, max_depth=12, n_jobs=-1,
    random_state=0, class_weight="balanced_subsample")
def hgb(): return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.03, max_iter=600,
    l2_regularization=2.0, random_state=0)
def cur_vote(): return VotingClassifier([("et", et()), ("hgb", hgb())], voting="soft", n_jobs=1)
def stack5():
    return StackingClassifier(
        estimators=[("lgb", lgbm()), ("xgb", xgbc()), ("cat", catb()), ("et", et()), ("rf", rf())],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        stack_method="predict_proba", cv=3, n_jobs=1, passthrough=False)


def evaluate_oof(name, mk, X, y, dates):
    t0 = time.time()
    oof = cross_val_predict(mk(), X, y, groups=dates, cv=GroupKFold(6),
                            method="predict_proba", n_jobs=1)[:, 1]
    ap = float(average_precision_score(y, oof))
    t = fpr_target_threshold(oof[y == 0], TARGET_FPR)
    rew, res = reward(remap_to_threshold(oof, t), y)
    log(f"{name:28} | AP={ap:.4f} reward={rew:.4f} fpr={res['fpr']:.3f} recall={res['bot_recall']:.3f}  ({time.time()-t0:.0f}s)")
    return ap, rew


if __name__ == "__main__":
    open(OUT, "w").write("# bakeoff_v3 (stacked ensemble vs current) on sanitized features_v2\n")
    if os.path.exists(CACHE):
        X, y, dates = load_cache(); log("# loaded cache")
    else:
        X, y, dates = build_cache()
    log(f"\n{'config':28} |  metrics")
    for name, mk in [
        ("CURRENT ET+HGB vote", cur_vote),
        ("LightGBM solo", lgbm),
        ("XGBoost solo", xgbc),
        ("CatBoost solo", catb),
        ("STACK5 -> LR meta", stack5),
    ]:
        try:
            evaluate_oof(name, mk, X, y, dates)
        except Exception as e:
            log(f"{name:28} | FAILED: {e}")
    log("# DONE")
