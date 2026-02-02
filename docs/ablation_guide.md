# Ablation and Benchmark Guide

## Baseline vs MCTS
MIO includes a baseline preset:
- text-like: WebP q=85
- photos: JPEG q=75 subsampling=4:2:0

Compare size reduction, SSIM, and runtime.

## Tile SSIM vs global SSIM
Compare tiles=1 vs tiles=4.

## Heuristics on/off
Disable text-like special rules to measure failure-rate changes.
