"""Poker44 ML miner entrypoint (lives in /root/model, runs against subnet 126).

Loads the trained RandomForest artifact and serves one calibrated bot-risk score
per chunk. Drop-in replacement for the repo's reference heuristic miner.

Run:
    POKER44_REPO=/root/pockcer44 python /root/model/poker44_miner.py \
        --netuid 126 --wallet.name my_cold --wallet.hotkey my_hot \
        --subtensor.network finney --axon.port 8091 \
        --blacklist.allowed_validator_hotkeys <val_hotkey_1> <val_hotkey_2>
"""

# NOTE: do NOT `from __future__ import annotations` here. bittensor's axon.attach
# introspects the real type of forward()'s `synapse` parameter via issubclass();
# stringized (PEP 563) annotations break that with "issubclass() arg 1 must be a
# class". The reference miner omits the future-import for the same reason.

import os
import sys
import time
from pathlib import Path
from typing import Tuple

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.environ.get("POKER44_REPO", "/root/pockcer44")
for p in (MODEL_DIR, REPO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (build_local_model_manifest,
                                          evaluate_manifest_compliance,
                                          manifest_digest)
from poker44.validator.synapse import DetectionSynapse

from infer import get_model


class MLMiner(BaseMinerNeuron):
    """Reference-compatible miner backed by the trained behavioral model."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.poker_model = get_model()
        meta = self.poker_model.meta
        self.model_manifest = build_local_model_manifest(
            repo_root=Path(MODEL_DIR),
            implementation_files=[
                Path(MODEL_DIR) / "infer.py",
                Path(MODEL_DIR) / "features_v2.py",
                Path(MODEL_DIR) / "artifacts" / "model.pkl",
                Path(MODEL_DIR) / "artifacts" / "meta.json",
            ],
            defaults={
                "model_name": os.environ.get("POKER44_MODEL_NAME", "poker44-d0"),
                "model_version": "2.0",
                "framework": "scikit-learn-extratrees-histgbm-ensemble",
                "license": "MIT",
                # TODO publish the model and set these to the real public repo/commit
                "repo_url": os.environ.get("POKER44_MODEL_REPO_URL",
                                           "https://github.com/<you>/poker44-miner"),
                "notes": (f"RandomForest on hero behavioral features. "
                          f"CV AP={meta['cv_ap']:.4f} reward={meta['cv_reward']:.4f} "
                          f"over {meta['n_dates']} benchmark dates."),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained only on the PUBLIC Poker44 benchmark "
                    "(api.poker44.net/api/v1/benchmark). No validator-only data used."),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": (
                    "This model does not train on validator-only evaluation data."),
                "data_attestation": (
                    "Features use only miner-visible behavioral fields; no cards, "
                    "board, outcome, or identifiers are used."),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        bt.logging.info(
            f"\U0001F9E0 Poker44 ML miner ready | cv_ap={meta['cv_ap']:.4f} "
            f"cv_reward={meta['cv_reward']:.4f} threshold={self.poker_model.threshold:.4f}")
        bt.logging.info(
            f"Manifest transparency: {self.manifest_compliance['status']} "
            f"(missing={self.manifest_compliance['missing_fields']}) "
            f"digest={manifest_digest(self.model_manifest)}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        try:
            scores = self.poker_model.score_chunks(chunks)
        except Exception as exc:  # never crash on a malformed request
            bt.logging.warning(f"scoring failed ({exc}); falling back to 0.5")
            scores = [0.5] * len(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Scored {len(chunks)} chunks | "
                        f"bots={sum(p for p in synapse.predictions)} "
                        f"mean={sum(scores)/max(len(scores),1):.3f}")
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with MLMiner() as miner:
        bt.logging.info("Poker44 ML miner running...")
        while True:
            try:
                bt.logging.info(
                    f"UID {miner.uid} | incentive {miner.metagraph.I[miner.uid]:.6f}")
            except Exception:
                pass
            time.sleep(5 * 60)
