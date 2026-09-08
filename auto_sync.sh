#!/bin/bash
cd ~/LoRA_Experiment
while true; do
  if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -m "Auto-sync: local update $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null 2>&1
  fi

  git pull >/tmp/sync_pull.log 2>&1
  if [ $? -ne 0 ]; then
    echo "[auto_sync] $(date -u +%H:%M:%S) conflict detected, resolving..."
    for f in $(git diff --name-only --diff-filter=U); do
      if [[ "$f" == *.jsonl ]]; then
        grep -v -E '^(<<<<<<<|=======|>>>>>>>)' "$f" > /tmp/merged_$$.jsonl
        mv /tmp/merged_$$.jsonl "$f"
        # validate every line is still parseable JSON before trusting the merge
        python3 -c "
import json, sys
bad = 0
with open('$f') as fh:
    for i, line in enumerate(fh, 1):
        line = line.strip()
        if not line: continue
        try: json.loads(line)
        except Exception: bad += 1
sys.exit(1 if bad else 0)
"
        if [ $? -eq 0 ]; then
          git add "$f"
          echo "[auto_sync] resolved $f (kept both sides, validated)"
        else
          echo "[auto_sync] WARNING: $f failed JSON validation after merge, needs manual look"
        fi
      fi
    done
    if [ -n "$(git diff --name-only --diff-filter=U)" ]; then
      echo "[auto_sync] UNRESOLVED conflict (non-jsonl or bad merge) - pausing 30s, check manually"
      sleep 30
      continue
    fi
    git commit -m "Auto-sync: merged concurrent pod results $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null 2>&1
  fi

  git push >/tmp/sync_push.log 2>&1 || echo "[auto_sync] $(date -u +%H:%M:%S) push rejected, retrying next cycle"

  sleep 60
done
