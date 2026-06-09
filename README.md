# DGM Spring 2026 — Face Image Generation

**Team:** Seonggyu_An  
**Final submission:** `mix_jpg95` (ADA snapshot-001895, JPEG q95)  
**Leaderboard result:** FID 29.23 / IS 4.68 / KID 0.0017 / TopPR 0.787

---

## Method Summary

StyleGAN2-ADA fine-tuned on CelebV-HQ (transfer from FFHQ-256 pretrained checkpoint),
with output images re-encoded as **JPEG quality 95** to match the JPEG encoding of the
reference dataset — the single most impactful improvement.

---

## Exact Seeds Used (Final Submission)

| Image range | Seeds | Truncation (ψ) |
|-------------|-------|----------------|
| img_0000 – img_0699 | 0 – 699 | 1.0 |
| img_0700 – img_0999 | 700 – 999 | 1.1 |

All images generated with `np.random.RandomState(seed)` — same checkpoint + seed + ψ
produces bit-identical PNG, then JPEG-encoded at quality 95.

---

## Environment

- Python 3.10
- PyTorch 2.6.0+cu124
- CUDA 12.4

Install dependencies:
```bash
pip install -r requirements.txt
```

Also requires **ninja** for StyleGAN2-ADA custom CUDA ops:
```bash
apt-get install ninja-build   # or: conda install ninja
```

---

## Setup

### 1. Clone StyleGAN2-ADA

```bash
git clone https://github.com/NVlabs/stylegan2-ada-pytorch external/stylegan2-ada-pytorch
```

### 2. Apply the PyTorch 2.6 compatibility patch

PyTorch 2.6 removed the double-backward path for `grid_sampler_2d_backward` needed by
ADA's R1 regularization. Without this patch, ADA silently falls back to no-augmentation
training.

```bash
cp patches/grid_sample_gradfix.py \
   external/stylegan2-ada-pytorch/torch_utils/ops/grid_sample_gradfix.py
```

### 3. Download the model checkpoint

Request `network-snapshot-001895.pkl` from the Top-10 submission (max 5 GB, shared via
email per course guidelines). Place it at:

```
checkpoints/network-snapshot-001895.pkl
```

---

## Inference — Reproduce Final Submission

```bash
bash scripts/generate_final.sh
```

This will:
1. Generate seeds 0–699 at ψ=1.0 → `outputs/final/seed0_699/`
2. Generate seeds 700–999 at ψ=1.1 → `outputs/final/seed700_999/`
3. Re-encode all PNGs as JPEG q95
4. Assemble `outputs/final/submission.zip` (1,000 images, named `img_XXXX.jpg`)

### Manual generation

```bash
# Seeds 0–699, ψ=1.0
python external/stylegan2-ada-pytorch/generate.py \
    --network checkpoints/network-snapshot-001895.pkl \
    --seeds 0-699 \
    --trunc 1.0 \
    --outdir outputs/final/seed0_699

# Seeds 700–999, ψ=1.1
python external/stylegan2-ada-pytorch/generate.py \
    --network checkpoints/network-snapshot-001895.pkl \
    --seeds 700-999 \
    --trunc 1.1 \
    --outdir outputs/final/seed700_999
```

### JPEG re-encoding

```bash
python scripts/png_to_jpeg.py \
    --input_dirs outputs/final/seed0_699 outputs/final/seed700_999 \
    --output_dir outputs/final/merged_jpg \
    --quality 95

python scripts/make_submission_zip.py \
    --input_dir outputs/final/merged_jpg \
    --output outputs/final/submission.zip
```

---

## Key Finding: JPEG Format Optimization

The reference CelebV-HQ images are stored as **JPEG**. Submitting PNG images creates a
systematic high-frequency mismatch in Inception-v3 feature space.  
Re-encoding generated images as **JPEG q95** (not q90, not raw PNG) moves generated
samples onto the same manifold as the reference set:

| Format | FID↓ | KID↓ | IS↑ |
|--------|------|------|-----|
| PNG (baseline) | 30.61 | 0.00413 | 4.49 |
| JPEG q90 | 30.57 | 0.00411 | 4.55 |
| **JPEG q95** | **29.28** | **0.00227** | 4.56 |

---

## Repository Structure

```
.
├── patches/
│   └── grid_sample_gradfix.py   # PyTorch 2.6 ADA compatibility patch
├── scripts/
│   ├── generate_final.sh        # End-to-end reproduction script
│   ├── png_to_jpeg.py           # PNG → JPEG q95 converter
│   └── make_submission_zip.py   # Assemble submission.zip
├── requirements.txt
└── README.md
```
