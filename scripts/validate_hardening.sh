#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/target/debug/momo"
PORT="${MOMO_PORT:-3000}"
BASE_URL="http://127.0.0.1:${PORT}"
API_KEY="${MOMO_VALIDATE_API_KEY:-validate-key}"
TEST_CONTAINER="${MOMO_VALIDATE_CONTAINER:-openclaw_vault}"
TMP_DIR="$(mktemp -d)"
LOG_FILE="${TMP_DIR}/momo.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" || true
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "[1/8] Build binary"
cargo build >/dev/null

echo "[2/8] Verify fail-closed startup (no API keys should fail)"
set +e
MOMO_ALLOW_NO_AUTH=0 MOMO_API_KEYS= MOMO_MCP_ENABLED=false "${BIN}" --mode api >/dev/null 2>&1
status=$?
set -e
if [[ ${status} -eq 0 ]]; then
  echo "FAIL: expected startup failure without API keys"
  exit 1
fi

echo "[3/8] Start hardened local server"
MOMO_API_KEYS="${API_KEY}" \
MOMO_HOST=127.0.0.1 \
MOMO_MCP_ENABLED=false \
MOMO_ENABLE_UPLOADS=false \
MOMO_CORS_ORIGINS=http://127.0.0.1:18888 \
"${BIN}" --mode api >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

for _ in {1..40}; do
  if curl -fsS -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/api/v1/health" >/dev/null; then
    break
  fi
  sleep 0.25
done

echo "[4/8] Health auth checks"
http_code="$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/api/v1/health")"
[[ "${http_code}" == "401" ]] || { echo "FAIL: expected 401 without auth, got ${http_code}"; exit 1; }
http_code="$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/api/v1/health")"
[[ "${http_code}" == "200" ]] || { echo "FAIL: expected 200 with auth, got ${http_code}"; exit 1; }

echo "[5/8] CORS allowlist check"
allow_origin="$(curl -s -I -X OPTIONS \
  -H "Origin: http://127.0.0.1:18888" \
  -H "Access-Control-Request-Method: GET" \
  "${BASE_URL}/api/v1/health" | tr -d '\r' | awk -F': ' '/^access-control-allow-origin:/ {print $2}')"
[[ "${allow_origin}" == "http://127.0.0.1:18888" ]] || { echo "FAIL: expected allowlisted CORS origin"; exit 1; }

echo "[6/8] Upload endpoint disabled check"
http_code="$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer ${API_KEY}" \
  -F "file=@${ROOT_DIR}/README.md" \
  "${BASE_URL}/api/v1/documents:upload")"
[[ "${http_code}" == "404" ]] || { echo "FAIL: expected 404 on disabled upload endpoint, got ${http_code}"; exit 1; }

echo "[7/8] documents:batch + search smoke test"
curl -fsS \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"containerTag\":\"${TEST_CONTAINER}\",\"documents\":[{\"content\":\"openclaw validation memory\"}]}" \
  "${BASE_URL}/api/v1/documents:batch" >/dev/null

sleep 1

curl -fsS \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"q\":\"openclaw validation\",\"containerTags\":[\"${TEST_CONTAINER}\"],\"scope\":\"hybrid\"}" \
  "${BASE_URL}/api/v1/search" >/dev/null

echo "[8/8] Loopback bind and log-safety checks"
if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN | grep -q "0.0.0.0:${PORT}"; then
  echo "FAIL: server is listening on 0.0.0.0:${PORT}"
  exit 1
fi

if grep -q "openclaw validation memory" "${LOG_FILE}"; then
  echo "FAIL: request body content leaked to logs"
  exit 1
fi

echo "PASS: hardening validation completed"
