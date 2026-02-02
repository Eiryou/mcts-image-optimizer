# SSIM Approximation

MIO uses a lightweight SSIM-like signal for search-time guidance.

## Steps
1) RGB conversion
2) Downscale to max side
3) Luma approximation: Y = 0.299R + 0.587G + 0.114B
4) Global SSIM approximation

Optional:
- 2×2 tiling (tiles=4)
- reduce_mode=min for text-like images (conservative)

## Limitations
- Not windowed SSIM
- Downscaling hides small artifacts
- Luma-only under-represents chroma artifacts
