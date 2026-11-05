#!/bin/bash
set -e

if ! command -v wasm-pack &> /dev/null; then
  echo "▸ installing wasm-pack..."
  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
  echo "▸ adding wasm target..."
  rustup target add wasm32-unknown-unknown
fi

echo "▸ building..."
wasm-pack build --target web --release

echo ""
echo "✓ build done"
echo "▸ serve: python3 -m http.server 8000"
echo "▸ open:  http://localhost:8000/"
