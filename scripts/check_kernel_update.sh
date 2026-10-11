#!/bin/bash
# Daily check for scmp_kernels updates. Appends status to ~/scmp_kernel_update.log.
# Installed in crontab; also safe to run by hand.
set -u
REPO=/home/qiuyid/scmp_worldmodel/kernels
LOG=/home/qiuyid/scmp_kernel_update.log
cd "$REPO" || { echo "$(date '+%F %T') ERROR: repo not found" >> "$LOG"; exit 1; }

git fetch origin -q 2>>"$LOG"
L=$(git rev-parse HEAD)
R=$(git rev-parse origin/main)
TS=$(date '+%F %T')

if [ "$L" != "$R" ]; then
    {
        echo "$TS UPDATE_AVAILABLE  local=$(git rev-parse --short HEAD)  remote=$(git rev-parse --short origin/main)"
        git log --oneline "HEAD..origin/main"
        # flag README/quant/sc changes that most likely need worldmodel re-alignment
        git diff --stat "HEAD..origin/main" -- README.md scmp_kernels/sc scmp_kernels/quant | sed 's/^/    /'
        echo "    -> pull with: git -C $REPO pull origin main   then re-check models/sc_integration alignment"
    } >> "$LOG"
else
    echo "$TS up-to-date ($(git rev-parse --short HEAD))" >> "$LOG"
fi
