#!/usr/bin/env bash
#
# Generate benchmark chart images and summary markdown from Hypura JSON results.
#
# Usage:
#   ./benchmarks/gen_charts.sh
#   ./benchmarks/gen_charts.sh benchmarks/results/*.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
CHARTS_DIR="$SCRIPT_DIR/charts"
OUTPUT="$SCRIPT_DIR/CHARTS.md"
GENERATOR="$SCRIPT_DIR/gen_charts.py"

if command -v python3 >/dev/null 2>&1 && python3 -c "import sys" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1 && python -c "import sys" >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "python3 or python is required"
    exit 1
fi

mkdir -p "$CHARTS_DIR"

if [ $# -gt 0 ]; then
    FILES=("$@")
else
    FILES=("$RESULTS_DIR"/*.json)
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No benchmark results found in $RESULTS_DIR"
    exit 1
fi

"$PYTHON_BIN" "$GENERATOR" --charts-dir "$CHARTS_DIR" --output-md "$OUTPUT" "${FILES[@]}"
