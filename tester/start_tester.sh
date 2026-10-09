#!/usr/bin/env bash
# start_tester.sh - start (or stop) the two 3LC services for testing the Kaggle Classification plugin
# on macOS or Linux.
#
#   ./start_tester.sh            # start (first run: logs you in to 3LC inside the tester folder)
#   ./start_tester.sh --stop     # stop everything it started
#
# Run it from the tester folder that holds the venv (TESTING.md step 2). Everything the services write
# lands under that folder: ./home (HOME for the services: the 3LC key you log in with, the compute
# service's settings, the plugin's venv and state, the downloaded kit, the prediction CSVs, the ledger)
# and ./logs. A Kaggle token under your real home is NOT seen by the services (HOME is redirected); the
# plugin's Connect button (TESTING.md 7.5b) writes the token it is given to ./home/.kaggle/access_token,
# nowhere else.
#
# Environment overrides (all optional):
#   OBJECT_PORT=5015       the 3LC object service port (the Getting Started page's default)
#   COMPUTE_PORT=5020      the compute service port (the Getting Started page's default)
#   PROJECT_ROOT=<dir>     keep the plugin's tables and runs in their own 3LC project root
#   MANIFEST_BASE=<url>    the competition manifest tier; default https://competitions.dev.3lc.ai
#   TEST_CATALOG=<url>     the plugin catalog listing the release candidate (default: the 1.0.0rc12 test catalog)
#   UV_CACHE_DIR=<dir>     reuse an existing uv cache (otherwise a fresh one under ./home)

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
STATE_HOME="$ROOT/home"
LOG_DIR="$ROOT/logs"
PID_FILE="$ROOT/start_tester.pids"
OBJECT_PORT="${OBJECT_PORT:-5015}"
COMPUTE_PORT="${COMPUTE_PORT:-5020}"
PROJECT_ROOT="${PROJECT_ROOT:-}"
MANIFEST_BASE="${MANIFEST_BASE:-https://competitions.dev.3lc.ai}"
TEST_CATALOG="${TEST_CATALOG:-https://raw.githubusercontent.com/3lc-ai/3lc-compute-plugin-kaggle-classification/release/1.0.0rc12/catalog-test.json}"
DEFAULT_CATALOG="https://3lc-public-examples-2-2.s3.amazonaws.com/hub/catalog.json"

say() { echo "$*"; echo "$*" >> "$ROOT/start_tester.last.log"; }

listening() {  # listening <port> -> 0 when something listens on 127.0.0.1:<port>
  if command -v lsof >/dev/null 2>&1; then lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; return $?; fi
  if command -v ss >/dev/null 2>&1; then ss -ltn 2>/dev/null | grep -q ":$1 "; return $?; fi
  (echo > "/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}

if [ "${1:-}" = "--stop" ]; then
  echo "stop_tester $(date -Iseconds)" > "$ROOT/start_tester.last.log"
  if [ -f "$PID_FILE" ]; then
    while read -r p; do
      [ -n "$p" ] && kill "$p" 2>/dev/null && say "Stopped pid $p"
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  fi
  # This folder's plugin workers and services only (matched on the folder path).
  pkill -f "tlc_plugin_sdk.worker.*$ROOT" 2>/dev/null && say "Stopped this folder's plugin workers"
  pkill -f "$VENV/bin/3lc-compute" 2>/dev/null; pkill -f "$VENV/bin/3lc service" 2>/dev/null
  sleep 1
  for port in "$OBJECT_PORT" "$COMPUTE_PORT"; do
    if listening "$port"; then say "Still listening on :$port"; exit 1; fi
  done
  say "Stopped: ports :$OBJECT_PORT and :$COMPUTE_PORT are free."
  exit 0
fi

echo "start_tester $(date -Iseconds)" > "$ROOT/start_tester.last.log"
for exe in 3lc 3lc-compute; do
  [ -x "$VENV/bin/$exe" ] || { say "Missing $VENV/bin/$exe - create the venv first (TESTING.md step 2)."; exit 1; }
done
command -v uv >/dev/null 2>&1 || { say "uv is not on PATH - the compute service needs it to build the plugin's environment (TESTING.md step 1)."; exit 1; }
for port in "$OBJECT_PORT" "$COMPUTE_PORT"; do
  if listening "$port"; then say "Already listening on :$port. Stop it (--stop), or set OBJECT_PORT / COMPUTE_PORT."; exit 1; fi
done
mkdir -p "$STATE_HOME" "$LOG_DIR"
[ -n "$PROJECT_ROOT" ] && mkdir -p "$PROJECT_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"

export HOME="$STATE_HOME"
export XDG_CONFIG_HOME="$STATE_HOME/.config"
export XDG_CACHE_HOME="$STATE_HOME/.cache"
[ -n "$PROJECT_ROOT" ] && export TLC_PROJECT_ROOT_URL="$PROJECT_ROOT"

# HOME is redirected, so the 3LC login lives inside the tester folder: log in once (interactive).
if [ ! -f "$STATE_HOME/.3lc-login-done" ]; then
  say "First start: logging in to 3LC inside $STATE_HOME (paste your API key from https://account.3lc.ai)."
  "$VENV/bin/3lc" login || { say "3LC login failed."; exit 1; }
  touch "$STATE_HOME/.3lc-login-done"
fi

run_loop() {  # run_loop <name> <command...>: restart on exit, everything to logs/<name>-<stamp>.log
  local name="$1"; shift
  local logf="$LOG_DIR/$name-$STAMP.log"
  local n=0
  while true; do
    n=$((n + 1))
    echo "[$(date -Iseconds)] start #$n - $name" >> "$logf"
    "$@" >> "$logf" 2>&1
    echo "[$(date -Iseconds)] $name EXITED with code $? - restarting in 5 s (run start_tester.sh --stop to stop)" >> "$logf"
    sleep 5
  done
}

( run_loop object "$VENV/bin/3lc" service --port "$OBJECT_PORT" --no-tui ) &
echo $! > "$PID_FILE"
say "Started the object service on :$OBJECT_PORT (log logs/object-$STAMP.log)"
( TLC_COMPUTE_PORT="$COMPUTE_PORT" \
  KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL="$MANIFEST_BASE" \
  TLC_COMPUTE_PLUGIN_CATALOG_URLS="$DEFAULT_CATALOG,$TEST_CATALOG" \
  run_loop compute "$VENV/bin/3lc-compute" ) &
echo $! >> "$PID_FILE"
say "Started the compute service on :$COMPUTE_PORT, manifest from $MANIFEST_BASE (log logs/compute-$STAMP.log)"

deadline=$(( $(date +%s) + 300 ))
ok_object=0; ok_compute=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  [ "$ok_object" = 0 ] && listening "$OBJECT_PORT" && { ok_object=1; say "  object :$OBJECT_PORT answers"; }
  [ "$ok_compute" = 0 ] && listening "$COMPUTE_PORT" && { ok_compute=1; say "  compute :$COMPUTE_PORT answers"; }
  [ "$ok_object" = 1 ] && [ "$ok_compute" = 1 ] && break
  sleep 2
done
if [ "$ok_object" != 1 ] || [ "$ok_compute" != 1 ]; then
  say "Not ready after 300 s: check logs/ for errors (a missing 3LC login is the usual cause)."
  exit 1
fi
say "Ready: open https://hub.3lc.ai/gettingstarted/ and connect the object service on 127.0.0.1:$OBJECT_PORT and the compute service on 127.0.0.1:$COMPUTE_PORT"
say "Plugin state and logs live under $ROOT (home/, logs/). Stop with: ./start_tester.sh --stop"
