#!/bin/bash
cd ~/LoRA_Experiment

resolve_conflicts() {
  local ok=1
  for f in $(git diff --name-only --diff-filter=U 2>/dev/null); do
    if [[ "$f" == *.jsonl ]]; then
      grep -v -E '^(<<<<<<<|=======|>>>>>>>)' "$f" > /tmp/merged_$$.jsonl
      mv /tmp/merged_$$.jsonl "$f"
      python3 -c "
import json,sys
bad=0
with open('$f') as fh:
    for l in fh:
        l=l.strip()
        if not l: continue
        try: json.loads(l)
        except Exception: bad+=1
sys.exit(1 if bad else 0)
" && git add "$f" || { echo "[auto_sync] $f failed validation"; ok=0; }
    else
      echo "[auto_sync] non-jsonl conflict in $f, needs manual look"
      ok=0
    fi
  done
  return $((1-ok))
}

while true; do
  if [ -f .git/MERGE_HEAD ] || git diff --name-only --diff-filter=U | grep -q .; then
    echo "[auto_sync] $(date -u +%H:%M:%S) leftover unresolved conflict, resolving first"
    if resolve_conflicts; then
      git commit -m "Auto-sync: merged concurrent pod results $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null 2>&1
    else
      echo "[auto_sync] could not auto-resolve, pausing 30s"; sleep 30; continue
    fi
  fi

  if [ -n "$(git status --porcelain)" ] && ! git diff --name-only --diff-filter=U | grep -q .; then
    git add -A
    git commit -m "Auto-sync: local update $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null 2>&1
  fi

  git pull >/tmp/sync_pull.log 2>&1
  if [ $? -ne 0 ]; then
    echo "[auto_sync] $(date -u +%H:%M:%S) pull conflict, resolving..."
    if resolve_conflicts; then
      git commit -m "Auto-sync: merged concurrent pod results $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null 2>&1
    else
      echo "[auto_sync] could not auto-resolve, pausing 30s"; sleep 30; continue
    fi
  fi

  git push >/tmp/sync_push.log 2>&1 || echo "[auto_sync] $(date -u +%H:%M:%S) push rejected, retry next cycle"
  sleep 60
done
