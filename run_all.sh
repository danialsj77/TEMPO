#!/usr/bin/env bash
# Final run of the TEMPO benchmark: every task, TEMPO and the five baselines, seeds 10-14, JOBS runs at a time (default 10).
#   bash run_all.sh                               # 10 runs at a time
#   JOBS=12 bash run_all.sh                       # 12 at a time
#   bash run_all.sh --dry-run                     # only show what would run
#   nohup bash run_all.sh > run_all.log 2>&1 &    # keep running after you log out (or use tmux / screen)
# Running it again resumes: finished runs are skipped. More options: python scripts/run_all.py --help
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python}" scripts/run_all.py --jobs "${JOBS:-10}" "$@"
