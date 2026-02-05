# MCTS Image Optimizer (MIO) — Streamlit Web Demo
> Note: If the input is already well-optimized, the reduction may be 0% (the original file is returned).

Search-based image compression demo: **MCTS × Heuristics × SSIM × Structural Intervention**.

This tool does not rely on fixed presets. Instead, it searches compression actions per image using:
- Monte Carlo Tree Search (MCTS)
- A scoring function combining size reduction, SSIM, and runtime
- Heuristic constraints (especially for text-like images)

## DOI(Zenodo)
10.5281/zenodo.18464299
https://doi.org/10.5281/zenodo.18464299

## DEMO URL
https://mcts-image-optimizer.onrender.com/

## X(Twitter)
https://x.com/nagisa7654321

## Qiita
https://qiita.com/Hideyoshi_Murakami/items/d63f136afe60f406d192


## Features
- Single image optimization (preview + metrics + download)
- Batch optimization (multiple images or ZIP input → ZIP output)
- Baseline comparison (fixed preset)
- Debug logs (optional)

## Disclaimer
This demo is provided “AS IS”, without warranty. Use at your own risk.  
Always keep backups of your original files.

**Author / Contact**: X (Twitter) **@nagisa7654321**

---
## License
Apache License 2.0

## Documentation (Technical Notes)

- `docs/overview.md` — big picture & design goals
- `docs/algorithm_mcts.md` — MCTS tree, selection/backprop details
- `docs/action_space.md` — why the action space is shaped this way
- `docs/scoring.md` — scoring function (math + tuning guide)
- `docs/ssim_approx.md` — SSIM approximation details and limitations
- `docs/heuristics.md` — text-like detection and safety constraints
- `docs/failure_modes.md` — known failure modes and why we guard them
- `docs/ablation_guide.md` — how to benchmark against baseline
- `docs/performance_render.md` — practical deployment tuning for Render
- `docs/deployment_render.md` — step-by-step Render deployment

---
##
Why so many clones but few stars?

This is a “try locally” type of project.
If this helped you, please consider starring ⭐ (it supports further development).

## Local Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open: http://localhost:8501

---

## Deploy on Render

This repository includes `render.yaml`, so you can deploy directly:
1. Push this repo to GitHub
2. Render → New → Web Service → Connect repo
3. Render detects `render.yaml`
4. Deploy

## Contact
For comments, work, and collaborations, please contact us here 
## murakami3tech6compres9sion@gmail.com   
