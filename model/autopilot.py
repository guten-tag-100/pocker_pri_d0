"""Autonomous training/deployment pilot for the Poker44 ML miner.

One command does the whole daily loop, safely:

  1. REFRESH  pull every available benchmark release and cache any missing dates
  2. RETRAIN  re-fit the model on all cached data (writes artifacts/ in place)
  3. GUARD    accept the new model ONLY if its honest cross-date CV reward does
              not regress and its human false-positive rate stays well under the
              reward() safety cliff (fpr >= 0.10 => reward 0). Otherwise revert.
  4. DEPLOY   if (and only if) a better model was promoted, restart the miner so
              it serves the new artifact.

Everything is idempotent and crash-safe: the previous artifact is backed up
before retraining and restored on any regression or failure, so the miner is
never left serving a worse (or broken) model.

Run manually:        python3 autopilot.py
Skip the restart:    python3 autopilot.py --no-restart
Force-deploy even
if reward ties/drops: python3 autopilot.py --force-deploy   (NOT recommended)

Scheduled daily by pm2 (see autopilot_schedule below) a few minutes after the
benchmark drops at 00:05 UTC.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
DATA = os.path.join(HERE, "data_cache")
BACKUPS = os.path.join(HERE, "artifacts_backups")
LOG = os.path.join(HERE, "autopilot.log")
HISTORY = os.path.join(HERE, "autopilot_history.jsonl")

API = "https://api.poker44.net/api/v1/benchmark"
PY = sys.executable  # use the SAME interpreter the miner runs under

# --- promotion guards ---------------------------------------------------------
# Accept a retrained model only if it does not meaningfully regress on honest
# cross-date CV and stays safely below the human-safety cliff.
REWARD_EPSILON = 0.002      # allow tiny noise; require new >= old - epsilon
MAX_DEPLOY_FPR = 0.06       # hard ceiling well under the reward() 0.10 cliff
# Name of the pm2 process that serves the model.
MINER_PM2_NAME = os.environ.get("POKER44_MINER_PM2", "poker44_miner")


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"{stamp} | {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "poker44-autopilot"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# 1. REFRESH
# ---------------------------------------------------------------------------
def refresh_data() -> int:
    """Download any benchmark dates we don't already have cached. Returns the
    number of NEW dates added this run."""
    os.makedirs(DATA, exist_ok=True)
    try:
        doc = json.loads(_get(f"{API}/releases?limit=100"))
        releases = [r["sourceDate"] for r in doc["data"]["releases"]]
    except Exception as exc:
        log(f"REFRESH: could not list releases ({exc}); using cached data only")
        return 0

    added = 0
    for d in sorted(releases):
        path = os.path.join(DATA, f"{d}.json")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        try:
            blob = _get(f"{API}/chunks?sourceDate={d}&limit=48")
            # validate it parses and has the expected shape before trusting it
            parsed = json.loads(blob)
            if "data" not in parsed or "chunks" not in parsed["data"]:
                log(f"REFRESH: {d} payload missing data.chunks; skipping")
                continue
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, path)
            added += 1
            log(f"REFRESH: cached new date {d}")
        except Exception as exc:
            log(f"REFRESH: failed to fetch {d} ({exc})")
    log(f"REFRESH: {added} new date(s); {len(os.listdir(DATA))} total cached")
    return added


# ---------------------------------------------------------------------------
# helpers for the guarded retrain
# ---------------------------------------------------------------------------
def read_meta(art_dir: str = ART) -> dict | None:
    p = os.path.join(art_dir, "meta.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def backup_current() -> str | None:
    """Snapshot the current artifact so we can revert. Returns backup dir."""
    if not os.path.exists(os.path.join(ART, "model.pkl")):
        return None
    os.makedirs(BACKUPS, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(BACKUPS, stamp)
    shutil.copytree(ART, dest)
    # keep only the 10 most recent backups
    snaps = sorted(d for d in os.listdir(BACKUPS)
                   if os.path.isdir(os.path.join(BACKUPS, d)))
    for old in snaps[:-10]:
        shutil.rmtree(os.path.join(BACKUPS, old), ignore_errors=True)
    return dest


def restore(backup_dir: str) -> None:
    shutil.rmtree(ART, ignore_errors=True)
    shutil.copytree(backup_dir, ART)


# ---------------------------------------------------------------------------
# 2 + 3. RETRAIN under guard
# ---------------------------------------------------------------------------
def retrain_and_guard(force_deploy: bool) -> bool:
    """Retrain in place but revert on regression/failure. Returns True if a NEW
    model was promoted (i.e. the served artifact changed for the better)."""
    old_meta = read_meta()
    old_reward = float(old_meta["cv_reward"]) if old_meta else -1.0
    old_dates = int(old_meta.get("n_dates", 0)) if old_meta else 0
    backup = backup_current()
    log(f"RETRAIN: baseline cv_reward={old_reward:.4f} over {old_dates} dates"
        f"{' (backed up)' if backup else ' (no prior model)'}")

    proc = subprocess.run([PY, os.path.join(HERE, "train_final_v2.py")],
                          cwd=HERE, capture_output=True, text=True,
                          env={**os.environ, "POKER44_REPO": "/root/my_pocker/pocker_d0",
                               "PYTHONUNBUFFERED": "1"})
    if proc.returncode != 0:
        log(f"RETRAIN: train_final.py FAILED rc={proc.returncode}")
        log(proc.stderr.strip()[-1500:])
        if backup:
            restore(backup)
            log("RETRAIN: reverted to previous artifact")
        return False

    new_meta = read_meta()
    if not new_meta:
        log("RETRAIN: no meta.json after training; reverting")
        if backup:
            restore(backup)
        return False

    new_reward = float(new_meta["cv_reward"])
    new_fpr = float(new_meta.get("cv_fpr", 1.0))
    new_dates = int(new_meta.get("n_dates", 0))
    log(f"RETRAIN: candidate cv_reward={new_reward:.4f} cv_fpr={new_fpr:.4f} "
        f"cv_ap={new_meta.get('cv_ap', 0):.4f} over {new_dates} dates")

    # --- promotion decision -------------------------------------------------
    reasons = []
    if new_fpr >= MAX_DEPLOY_FPR:
        reasons.append(f"fpr {new_fpr:.4f} >= ceiling {MAX_DEPLOY_FPR}")
    if not force_deploy and new_reward < old_reward - REWARD_EPSILON:
        reasons.append(f"reward {new_reward:.4f} < baseline {old_reward:.4f} - eps")

    if reasons and backup:
        log("RETRAIN: REJECTED (" + "; ".join(reasons) + ") -> reverting")
        restore(backup)
        record_history("rejected", old_reward, new_reward, new_fpr, new_dates, reasons)
        return False
    if reasons and not backup:
        # first ever train but it violates the fpr ceiling: keep it (better than
        # nothing) but shout about it.
        log("RETRAIN: WARNING first model violates a guard but no backup to "
            "revert to: " + "; ".join(reasons))

    improved = new_reward > old_reward + REWARD_EPSILON
    verb = "PROMOTED (improved)" if improved else "PROMOTED (>= baseline)"
    log(f"RETRAIN: {verb} cv_reward {old_reward:.4f} -> {new_reward:.4f}")
    record_history("promoted", old_reward, new_reward, new_fpr, new_dates, [])
    # Restart whenever the served artifact meaningfully changed: either reward
    # improved, or new data was incorporated (more dates => fresher model).
    return improved or new_dates > old_dates


def record_history(decision, old_reward, new_reward, fpr, n_dates, reasons):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "old_reward": old_reward,
        "new_reward": new_reward,
        "fpr": fpr,
        "n_dates": n_dates,
        "reasons": reasons,
    }
    with open(HISTORY, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# 4. DEPLOY
# ---------------------------------------------------------------------------
def restart_miner() -> None:
    """Reload the miner pm2 process so it picks up the new artifact."""
    try:
        out = subprocess.run(["pm2", "restart", MINER_PM2_NAME, "--update-env"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            log(f"DEPLOY: restarted pm2 process '{MINER_PM2_NAME}'")
        else:
            log(f"DEPLOY: pm2 restart failed rc={out.returncode}: "
                f"{out.stderr.strip()[-400:]}")
    except FileNotFoundError:
        log("DEPLOY: pm2 not found on PATH; restart the miner manually")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-restart", action="store_true",
                    help="retrain/promote but do not restart the miner")
    ap.add_argument("--force-deploy", action="store_true",
                    help="promote even if reward ties/regresses (still honors fpr ceiling)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip the data download step")
    args = ap.parse_args()

    t0 = time.time()
    log("=== AUTOPILOT START ===")
    added = 0 if args.no_refresh else refresh_data()
    promoted = retrain_and_guard(force_deploy=args.force_deploy)
    if promoted and not args.no_restart:
        restart_miner()
    elif promoted:
        log("DEPLOY: promotion happened but --no-restart set; serving stale until restart")
    else:
        log("DEPLOY: nothing to deploy (model unchanged)")
    log(f"=== AUTOPILOT DONE in {time.time()-t0:.0f}s | new_dates={added} "
        f"promoted={promoted} ===")


if __name__ == "__main__":
    main()
