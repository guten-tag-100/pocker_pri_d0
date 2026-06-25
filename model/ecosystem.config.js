// pm2 process definitions for the Poker44 ML miner + autonomous training pilot.
//
//   pm2 start /root/pocker/model/ecosystem.config.js
//   pm2 save            # persist across reboots
//
// App 1 (poker44_miner)    : serves the trained model's calibrated risk scores.
// App 2 (poker44_autopilot): runs once daily at 00:10 UTC to refresh the
//                            benchmark, retrain under guard, and (if the model
//                            improved) restart the miner. autorestart:false +
//                            cron_restart makes pm2 treat it as a scheduled job.

const { execSync } = require("child_process");

const PY = "/root/pocker/miner_env/bin/python";   // env that has bittensor + sklearn
const REPO = "/root/pocker";                       // the actual subnet repo
const MODEL = "/root/pocker/model";

// Report the ACTUAL git HEAD of the repo as the manifest repo_commit, so the
// published commit always matches the served code (no hardcoded-hash drift,
// no self-referential chicken-and-egg). Whatever commit is checked out + pushed
// is what the miner advertises.
let REPO_COMMIT = "";
try { REPO_COMMIT = execSync(`git -C ${REPO} rev-parse HEAD`).toString().trim(); }
catch (e) { REPO_COMMIT = ""; }

module.exports = {
  apps: [
    {
      name: "poker44_miner",
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
        // public model identity for the compliance manifest (must match served files)
        POKER44_MODEL_REPO_URL: "https://github.com/guten-tag-100/pocker44-v1",
        POKER44_MODEL_REPO_COMMIT: REPO_COMMIT,
      },
      autorestart: true,
      max_restarts: 20,
      min_uptime: "30s",
      restart_delay: 5000,
      kill_timeout: 10000,
    },
    {
      name: "poker44_autopilot",
      script: `${MODEL}/autopilot.py`,
      interpreter: PY,
      cwd: MODEL,
      env: { POKER44_REPO: REPO, POKER44_MINER_PM2: "poker44_miner" },
      autorestart: false,        // run-to-completion, not a daemon
      cron_restart: "10 0 * * *", // 00:10 UTC daily, just after the 00:05 drop
    },
  ],
};
