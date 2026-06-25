# Poker44 Miner — behavioral bot-detection (v2)

Miner for **Poker44, Bittensor subnet 126**. The validator sends
`DetectionSynapse(chunks=...)`; each chunk is ~30–90 poker hands from ONE hero
player. We return **one bot-risk score in [0,1] per chunk** (high = bot-like),
purely from betting behavior — we never see cards, board, or outcome.

## Why v2 (the key idea)

The validator does **not** send raw hands. Every hand is passed through
`poker44.validator.payload_view.prepare_hand_for_miner`, which:

- re-aliases seats (so absolute `hero_seat` / button position is meaningless),
  forces `button_seat = 0`;
- normalizes blinds to `sb=0.01 / bb=0.02` and converts amounts to bb;
- **coarsens every bet amount into 16 fixed bb-buckets with hash noise**;
- keeps only a **random 5–8 action window per hand**, so the hero frequently has
  zero visible actions.

A model that keys off the hero's actions or raw amounts/positions trains fine on
raw benchmark data but **collapses on the live feed**. v2 fixes this two ways:

1. **`features_v2.py`** — 250 hero-free, sanitization-invariant features:
   per-hand action-mix shares, entropies and run/switch regularity over **all**
   actions; bet sizes **quantized to the validator's exact bb-bucket grid**;
   cross-hand **signature** features (`*_top_share` / `*_unique_share` over
   action/role/street/amount-bucket sequences) that catch bots replaying near
   identical hands; everything aggregated to chunk level with order-stats
   (mean/std/min/max/q10/q50/q90) so 30-hand and 85-hand chunks look alike.
2. **train == serve** — every training chunk is run through
   `prepare_hand_for_miner` before featurizing, so the model sees the same
   distribution it serves.

Model: a soft-voting **ExtraTrees + HistGradientBoosting** ensemble
(scikit-learn), with a conservative conformal deploy threshold chosen from
cross-date out-of-fold human scores to stay under the reward's human-safety FPR
cliff (fpr ≥ 0.10 → reward 0).

## Measured performance (honest cross-date CV on the SANITIZED / live-equivalent feed)

| metric | v2 | (old v1, sanitized) |
|---|---|---|
| Average Precision (AP) | **0.880** | 0.745 |
| reward() (repo's exact fn) | **0.705** | 0.450 |
| human FPR | 0.032 | 0.037 |
| bot recall | 0.51 | 0.24 |

Evaluated with `GroupKFold` by release date (never tested on a trained date) and
the repo's exact `reward()`.

## Files

| file | purpose |
|---|---|
| `features_v2.py` | chunk → 250 hero-free, sanitization-invariant features (train == serve) |
| `dataset.py` | load cached benchmark releases → labeled chunk groups |
| `train_final_v2.py` | sanitize → featurize → fit ET+HGB ensemble → `artifacts/` |
| `evaluate.py` / `reward_fn.py` | cross-date CV harness + the repo's reward() |
| `infer.py` | `Poker44Model.score_chunks(chunks) -> [risk_score]` (serving) |
| `poker44_miner.py` | miner entrypoint (subclasses repo `BaseMinerNeuron`) |
| `autopilot.py` | daily refresh + guarded retrain + redeploy |
| `ecosystem.config.js` | pm2 process defs (miner + daily autopilot) |
| `artifacts/` | `model.pkl` + `meta.json` (feature order, threshold, CV metrics) |

## Retrain

```bash
cd /root/model
# 1. refresh benchmark cache (daily drop ~00:05 UTC)
#    autopilot.py does this automatically; or fetch into data_cache/ manually.
# 2. retrain on the SANITIZED payload and save artifacts/
POKER44_REPO=/root/pocker python3 train_final_v2.py
```

## Run

```bash
POKER44_REPO=/root/pocker pm2 start ecosystem.config.js && pm2 save
```

`autopilot.py` runs daily (00:10 UTC via pm2 cron): it refreshes the benchmark,
retrains under guard (only promotes a model whose cross-date CV reward does not
regress and whose FPR stays under the cliff), and restarts the miner if the model
improved.
