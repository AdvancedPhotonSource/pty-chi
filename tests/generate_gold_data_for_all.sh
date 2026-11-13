#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for test_file in test_*_e2e.py; do
    if ! grep -q "wrap_recon_tester" "$test_file"; then
        continue
    fi
    if grep -q '@pytest.mark.local' "$test_file"; then
        continue
    fi
    python "$test_file" --generate-gold
done
