#!/usr/bin/env bash
#
# run_detached.sh <run_id> <command...>
#
# Launch a long-running command DETACHED from the current shell/session so it
# survives the caller dying -- specifically, a killed Claude Code session.
#
# Why setsid: a normal backgrounded child stays in the launching process's
# process group and receives SIGHUP when that session ends, taking hours of
# pipeline work down with it. `setsid` puts the job in a brand-new session with
# no controlling terminal, so it is immune to that hangup.
#
# The wrapper is deliberately DUMB: it runs the given command verbatim and does
# nothing else. That is the non-breaking guarantee -- the pipeline writes the
# same caches (table cache, eval-summary cache) and the same version-folder
# outputs whether it is detached or run by hand.
#
# Per-run state lands in .runs/<run_id>/ (gitignored):
#   cmd         the command, re-runnable
#   started_at  UTC ISO-8601
#   ended_at    UTC ISO-8601 (only if the process exits normally)
#   pid         session-leader pid
#   status      RUNNING -> "<exit-code>"  (stays RUNNING if killed: that is the
#               signal that it died without finishing)
#   log         merged stdout+stderr
#
# Monitor with:  scripts/job_status.sh <run_id>
set -euo pipefail

RUN_ID="${1:-}"
if [ -z "$RUN_ID" ] || [ "$#" -lt 2 ]; then
  echo "usage: scripts/run_detached.sh <run_id> <command...>" >&2
  exit 2
fi
shift

# Reject ids that would escape the .runs/ dir.
case "$RUN_ID" in
  */*|.*) echo "run_id must be a simple name (no '/' or leading '.')" >&2; exit 2;;
esac

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
DIR="$ROOT/.runs/$RUN_ID"
if [ -e "$DIR" ]; then
  echo "run id '$RUN_ID' already exists at $DIR -- pick a fresh id" >&2
  exit 3
fi
mkdir -p "$DIR"

# Record the command as a re-runnable, safely-quoted string.
printf '%q ' "$@" > "$DIR/cmd"; printf '\n' >> "$DIR/cmd"
date -u +%Y-%m-%dT%H:%M:%SZ > "$DIR/started_at"
printf 'RUNNING\n' > "$DIR/status"

# Launch. The inner script runs the command, then records its exit code and end
# time. If the job is hard-killed, these final lines never run and `status`
# stays RUNNING -- job_status.sh reads that (pid dead + status RUNNING) as DIED.
setsid bash -c '
  "$@"
  code=$?
  printf "%s\n" "$code" > "'"$DIR"'/status"
  date -u +%Y-%m-%dT%H:%M:%SZ > "'"$DIR"'/ended_at"
' _ "$@" > "$DIR/log" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$DIR/pid"
disown "$PID" 2>/dev/null || true

echo "started run '$RUN_ID' (pid $PID) -- detached, survives this session"
echo "  log:    .runs/$RUN_ID/log"
echo "  status: .runs/$RUN_ID/status"
echo "  watch:  scripts/job_status.sh $RUN_ID"
