#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 --version
echo "Runner version:"
python3 run_all.py --version || true
echo
python3 run_all.py       --show-static-scan --show-tamper --show-bridge       --density-cert --density-cert-v2 --compute-scale       --use-solved-scale       --json examples/prooflog_example.json
echo
echo "Wrote examples/prooflog_example.json"
