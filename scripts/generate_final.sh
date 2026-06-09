#!/usr/bin/env bash
# Reproduce the final submission:
#   seeds 0-699  at psi=1.0  +  seeds 700-999 at psi=1.1
#   all re-encoded as JPEG quality 95
#
# Usage:
#   bash scripts/generate_final.sh [path/to/network-snapshot-001895.pkl]
#
# Prerequisites:
#   1. pip install -r requirements.txt
#   2. apt-get install ninja-build   (or: conda install ninja)
#   3. git clone https://github.com/NVlabs/stylegan2-ada-pytorch external/stylegan2-ada-pytorch
#   4. Place network-snapshot-001895.pkl under checkpoints/

set -euo pipefail

NETWORK="${1:-checkpoints/network-snapshot-001895.pkl}"
STYLEGAN_DIR="external/stylegan2-ada-pytorch"
OUT_BASE="outputs/final"

# ── Preflight checks ──────────────────────────────────────────────────────────

if [[ ! -f "$NETWORK" ]]; then
  echo "ERROR: checkpoint not found at '$NETWORK'"
  echo "  Place network-snapshot-001895.pkl at that path and re-run."
  exit 1
fi

if [[ ! -f "$STYLEGAN_DIR/generate.py" ]]; then
  echo "ERROR: stylegan2-ada-pytorch not found at '$STYLEGAN_DIR'"
  echo "  Run: git clone https://github.com/NVlabs/stylegan2-ada-pytorch $STYLEGAN_DIR"
  exit 1
fi

# ── Step 0: Apply PyTorch 2.6 compatibility patch ────────────────────────────
echo "=== Step 0/4: Applying PyTorch 2.6 compatibility patch ==="
PATCH_DST="$STYLEGAN_DIR/torch_utils/ops/grid_sample_gradfix.py"
cp patches/grid_sample_gradfix.py "$PATCH_DST"
echo "    Patched: $PATCH_DST"

# ── Step 1: Generate seeds 0-699 at psi=1.0 ──────────────────────────────────
echo "=== Step 1/4: Generating seeds 0-699 (truncation psi=1.0) ==="
python "$STYLEGAN_DIR/generate.py" \
  --network "$NETWORK" \
  --seeds 0-699 \
  --trunc 1.0 \
  --outdir "$OUT_BASE/psi1.0"

# ── Step 2: Generate seeds 700-999 at psi=1.1 ────────────────────────────────
echo "=== Step 2/4: Generating seeds 700-999 (truncation psi=1.1) ==="
python "$STYLEGAN_DIR/generate.py" \
  --network "$NETWORK" \
  --seeds 700-999 \
  --trunc 1.1 \
  --outdir "$OUT_BASE/psi1.1"

# ── Step 3: Re-encode PNG → JPEG quality 95 ──────────────────────────────────
echo "=== Step 3/4: Re-encoding PNG images to JPEG quality 95 ==="
python scripts/png_to_jpeg.py \
  --input_dirs "$OUT_BASE/psi1.0" "$OUT_BASE/psi1.1" \
  --output_dir "$OUT_BASE/jpeg_merged" \
  --quality 95

# ── Step 4: Build submission ZIP ──────────────────────────────────────────────
echo "=== Step 4/4: Building submission.zip ==="
python scripts/make_submission_zip.py \
  --input-dir "$OUT_BASE/jpeg_merged" \
  --output    "$OUT_BASE/submission.zip"

echo ""
echo "Done. Final submission: $OUT_BASE/submission.zip"
