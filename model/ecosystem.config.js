// pm2 process defs for the pocker_d0 miner (wallet dragon / hotkey dragon0, UID 92).
const { execSync } = require("child_process");
const PY = "/root/my_pocker/pocker_d0/miner_env/bin/python";
const REPO = "/root/my_pocker/pocker_d0";
const MODEL = "/root/my_pocker/pocker_d0/model";
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
        "--netuid", "126",
        "--wallet.name", "dragon",
        "--wallet.hotkey", "dragon0",
        "--subtensor.network", "finney",
        "--axon.port", "8091",
        "--logging.debug",
        "--blacklist.force_validator_permit",
      ].join(" "),
      env: {
        POKER44_REPO: REPO,
        POKER44_MODEL_REPO_URL: "https://github.com/guten-tag-100/pocker44-v1",
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
