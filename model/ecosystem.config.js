// pm2 process defs for the pocker_d0 miner.
// SECURITY: no wallet/hotkey/ports are hardcoded here (this file is committed to
// git). All private/operational config is read from `.env` (gitignored). Copy
// `.env.example` -> `.env` and fill it in before running.
const fs = require("fs");
const { execSync } = require("child_process");

const REPO = "/root/my_pocker/pocker_d0";
const MODEL = `${REPO}/model`;
const PY = `${REPO}/miner_env/bin/python`;

function loadEnv(p) {
  const out = {};
  try {
    for (const raw of fs.readFileSync(p, "utf8").split("\n")) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const i = line.indexOf("=");
      if (i > 0) out[line.slice(0, i).trim()] = line.slice(i + 1).trim().replace(/^["']|["']$/g, "");
    }
  } catch (e) {}
  return out;
}
const E = { ...loadEnv(`${REPO}/.env`), ...process.env };

const WALLET = E.POKER44_WALLET_NAME;
const HOTKEY = E.POKER44_WALLET_HOTKEY;
const NETUID = E.POKER44_NETUID || "126";
const PORT = E.POKER44_AXON_PORT || "8091";
const REPO_URL = E.POKER44_MODEL_REPO_URL || "";
if (!WALLET || !HOTKEY) {
  throw new Error("pocker_d0: missing POKER44_WALLET_NAME / POKER44_WALLET_HOTKEY — create .env from .env.example");
}

let REPO_COMMIT = "";
try { REPO_COMMIT = execSync(`git -C ${REPO} rev-parse HEAD`).toString().trim(); } catch (e) {}

module.exports = {
  apps: [
    {
      name: "poker44_miner_d0",
      script: `${MODEL}/poker44_miner.py`,
      interpreter: PY,
      cwd: REPO,
      args: [
        "--netuid", NETUID,
        "--wallet.name", WALLET,
        "--wallet.hotkey", HOTKEY,
        "--subtensor.network", "finney",
        "--axon.port", PORT,
        "--logging.debug",
        "--blacklist.force_validator_permit",
      ].join(" "),
      env: {
        POKER44_REPO: REPO,
        POKER44_MODEL_REPO_URL: REPO_URL,
        POKER44_MODEL_REPO_COMMIT: REPO_COMMIT,
      },
      autorestart: true, max_restarts: 20, min_uptime: "30s",
      restart_delay: 5000, kill_timeout: 10000,
    },
    {
      name: "poker44_autopilot_d0",
      script: `${MODEL}/autopilot.py`,
      interpreter: PY,
      cwd: MODEL,
      env: { POKER44_REPO: REPO, POKER44_MINER_PM2: "poker44_miner_d0" },
      autorestart: false,
      cron_restart: "10 0 * * *",
    },
  ],
};
