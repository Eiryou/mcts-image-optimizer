# Heuristics and Constraints

## Text-like detection (heuristic)
Uses:
- edge density
- colorfulness
- estimated colors

Rule-of-thumb thresholds are used (not a trained classifier).

## Why special handling?
Text-like inputs can become unreadable quickly.
MIO:
- raises SSIM threshold
- avoids aggressive JPEG subsampling+low quality
- blocks excessive downscale for text-like images
