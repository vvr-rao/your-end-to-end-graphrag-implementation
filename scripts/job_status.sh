#!/usr/bin/env bash
#
# job_status.sh [run_id] [n_log_lines]
#
# Report on a detached run started by run_detached.sh. With no run_id, lists
# every run and its state. This is how Claude re-attaches after a session death:
# it does not hold the process, it reads this.
#
# State is derived from (pid alive?) x (status file):
#   status "RUNNING" + pid alive  -> RUNNING
#   status "RUNNING" + pid dead   -> DIED   (killed/crashed before finishing)
#   status "0"                    -> SUCCESS
#   status "<non-zero>"           -> FAILED (exit <code>)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
RUNS="$ROOT/.runs"

_state() {  # $1=dir -> echoes STATE; sets no globals
  local dir="$1" pid status live
  pid="$(cat "$dir/pid" 2>/dev/null || echo '')"
  status="$(cat "$dir/status" 2>/dev/null || echo '?')"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then live=alive; else live=dead; fi
  if   [ "$status" = RUNNING ] && [ "$live" = alive ]; then echo "RUNNING"
  elif [ "$status" = RUNNING ] && [ "$live" = dead  ]; then echo "DIED (no exit code — killed/crashed)"
  elif [ "$status" = 0 ]; then echo "SUCCESS"
  elif [ "$status" = '?' ]; then echo "UNKNOWN (no status file)"
  else echo "FAILED (exit $status)"; fi
}

RUN_ID="${1:-}"

# No id: list all runs.
if [ -z "$RUN_ID" ]; then
  if [ ! -d "$RUNS" ] || [ -z "$(ls -A "$RUNS" 2>/dev/null)" ]; then
    echo "no runs yet (.runs/ is empty)"
    exit 0
  fi
  printf '%-24s %-8s %s\n' "RUN" "PID" "STATE"
  for d in "$RUNS"/*/; do
    [ -d "$d" ] || continue
    id="$(basename "$d")"
    printf '%-24s %-8s %s\n' "$id" "$(cat "$d/pid" 2>/dev/null || echo '?')" "$(_state "$d")"
  done
  exit 0
fi

DIR="$RUNS/$RUN_ID"
[ -d "$DIR" ] || { echo "no such run: $RUN_ID" >&2; exit 1; }

PID="$(cat "$DIR/pid" 2>/dev/null || echo '?')"
if [ "$PID" != '?' ] && kill -0 "$PID" 2>/dev/null; then LIVE=alive; else LIVE=dead; fi
LINES="${2:-30}"

echo "run:     $RUN_ID"
echo "state:   $(_state "$DIR")"
echo "pid:     $PID ($LIVE)"
echo "cmd:     $(cat "$DIR/cmd" 2>/dev/null || echo '?')"
echo "started: $(cat "$DIR/started_at" 2>/dev/null || echo '?')"
[ -f "$DIR/ended_at" ] && echo "ended:   $(cat "$DIR/ended_at")"
echo "----- last $LINES log lines (.runs/$RUN_ID/log) -----"
tail -n "$LINES" "$DIR/log" 2>/dev/null || echo "(no log yet)"
