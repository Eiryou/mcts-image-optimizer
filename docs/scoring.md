# Scoring Function (Math + Tuning)

## Hard constraints
A candidate is rejected if:
- output size grows beyond allowed percent (default 0%)
- output cannot be decoded
- SSIM < target (text-like raises threshold to at least 0.975)

## Definitions
Let S_orig be original bytes, S_out output bytes.

Reduction ratio:
r = (S_orig - S_out) / S_orig

SSIM margin:
m = SSIM - target

Runtime:
tau = milliseconds

## Score
score = w_s*(1000*r) + w_q*(1000*m) - w_t*tau

### Tuning
- smaller files: increase size weight, lower target SSIM slightly
- safer quality: increase target SSIM and SSIM weight
- faster: lower budget, lower ssim_max_side, increase time penalty
