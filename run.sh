#!/usr/bin/env bash
# Run the full pipeline: train → evaluate
# Usage: ./run.sh [optional args passed to both commands]
# Example: ./run.sh --epochs 20 --batch-size 32

set -e  # exit on any error

echo ""
echo "=================================================="
echo " FER-AffectNet  |  Full Pipeline"
echo "=================================================="

echo ""
echo ">>> STEP 1/2: Training"
echo "--------------------------------------------------"
uv run fer-train "$@"

echo ""
echo ">>> STEP 2/2: Evaluation"
echo "--------------------------------------------------"
uv run fer-evaluate "$@"

echo ""
echo "=================================================="
echo " Done. Outputs saved to: outputs/"
echo "=================================================="
