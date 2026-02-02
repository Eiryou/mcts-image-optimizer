# Failure Modes and Mitigations

1) Output larger than input
- blocked by default to stabilize search and user expectations

2) Decode failure
- decode verification is mandatory

3) Text unreadable
- text-like heuristics enforce conservative constraints

4) SSIM passes but perceptual quality is worse
- SSIM approximation limitations; tiled SSIM and conservative modes help

5) Slow candidates on small servers
- per-action timeout + global safety break + limited action space
