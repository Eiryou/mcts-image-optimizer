# Performance Tuning for Render

Key knobs:
- budget
- per_action_timeout_s
- ssim_max_side

## Safe free-tier suggestion
- budget: ~80
- ssim_max_side: 384–512
- tiles: 1 if slow
- per_action_timeout_s: 0.8–1.2
