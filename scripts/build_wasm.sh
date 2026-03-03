#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="${ROOT_DIR}/web"

mkdir -p "${WEB_DIR}"
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" "${WEB_DIR}/wasm_exec.js"

cd "${ROOT_DIR}"
GOOS=js GOARCH=wasm go build -o "${WEB_DIR}/main.wasm" .

echo "WASM build complete:"
echo "  ${WEB_DIR}/main.wasm"
echo "  ${WEB_DIR}/wasm_exec.js"
echo "Open with a static server, for example:"
echo "  cd ${WEB_DIR} && python3 -m http.server 8080"
