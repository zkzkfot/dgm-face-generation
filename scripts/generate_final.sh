#!/usr/bin/env bash
# Reproduce the final submission (mix_jpg95):
#   seeds 0-699  @ psi=1.0  +  seeds 700-999 @ psi=1.1
#   all re-encoded as JPEG quality 95
# Usage: bash scripts/generate_final.sh [path/to/network-snapshot-001895.pkl]
set -euo pipefail

NETWORK="${1:-checkpoints/network-snapshot-001895.pkl}"
STYLEGAN_DIR="external/stylegan2-ada-pytorch"
OUT_BASE="outputs/final"

if [[ ! -f "$NETWORK" ]]; then
  echo "ERROR: checkpoint not found at $NETWORK"
  echo "Place network-snapshot-001895.pkl at $NETWORK and re-run."
  exit 1
fi

if [[ ! -f "$STYLEGAN_DIR/generate.py" ]]; then
  echo "ERROR: stylegan2-ada-pytorch not found."
  echo "Run: git clone https://github.com/NVlabs/stylegan2-ada-pytorch $STYLEGAN_DIR"
  exit 1
fi

echo "=== Step 1/4: Generate seeds 0-699 (psi=1.0) ==="
python "$STYLEGAN_DIR/generate.py" \
  --network "$NETWORK" \
  --seeds 0-699 \
  --trunc 1.0 \
  --outdir "$OUT_BASE/seed0_699"

echo "=== Step 2/4: Generate seeds 700-999 (psi=1.1) ==="
python "$STYLEGAN_DIR/generate.py" \
  --network "$NETWORK" \
  --seeds 700-999 \
  --trunc 1.1 \
  --outdir "$OUT_BASE/seed700_999"

echo "=== Step 3/4: Re-encode PNG -> JPEG q95 ==="
python scripts/png_to_jpeg.py \
  --input_dirs "$OUT_BASE/seed0_699" "$OUT_BASE/seed700_999" \
  --output_dir "$OUT_BASE/merged_jpg" \
  --quality 95

echo "=== Step 4/4: Build submission.zip ==="
python scripts/make_submission_zip.py \
  --input_dir "$OUT_BASE/merged_jpg" \
  --output "$OUT_BASE/submission.zip"

echo ""
echo "Done. Submission zip: $OUT_BASE/submission.zip"
python -c "
import zipfile, os
z = '$OUT_BASE/submission.zip'
with zipfile.ZipFile(z) as zf:
    n = len(zf.namelist())
size_mb = os.path.getsize(z) / 1024**2
print(f'  {n} images, {size_mb:.1f} MB')
"
