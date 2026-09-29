#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${1:-http://127.0.0.1:8000}
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

ready=false
for _ in {1..30}; do
    if curl --fail --silent --max-time 2 "$BASE_URL/api/config" -o "$TMP_DIR/config.json"; then
        ready=true
        break
    fi
    sleep 1
done
if [[ "$ready" != true ]]; then
    echo "MetroGuard did not become ready at $BASE_URL" >&2
    exit 1
fi
python3 - "$TMP_DIR/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    payload = json.load(source)
assert payload["best_algorithm"] in payload["algorithms"]
assert payload["max_upload_bytes"] == 8 * 1024**3
PY

curl --fail --silent --show-error --max-time 15 "$BASE_URL/metrics" -o /dev/null
printf 'not sqlite' >"$TMP_DIR/invalid.db3"
status=$(curl --silent --show-error --max-time 15 -o "$TMP_DIR/rejection.json" -w '%{http_code}' \
    -F "algorithm=best" -F "bag=@$TMP_DIR/invalid.db3;type=application/octet-stream" \
    "$BASE_URL/api/jobs")
if [[ "$status" != 400 ]]; then
    echo "expected invalid upload to return 400, got $status" >&2
    cat "$TMP_DIR/rejection.json" >&2
    exit 1
fi

echo "MetroGuard smoke test passed: $BASE_URL"
