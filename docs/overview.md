# Overview

**MCTS Image Optimizer (MIO)** is a Streamlit demo that optimizes image compression through *search*, not presets.

> Image compression is a combinatorial decision problem (codec × quality × resize × filters).
> The best choice is highly input-dependent (photos vs screenshots vs scans).
> Therefore, a search procedure guided by a quality metric can outperform “one-size-fits-all” presets.

MIO implements that idea in a practical form:
- Action candidates are generated from a lightweight **image profile**
- Each action is evaluated by a **multi-objective score**:
  - size reduction (primary goal)
  - SSIM (quality constraint + margin bonus)
  - runtime penalty (deployment friendliness)
- Search is performed using **Monte Carlo Tree Search (MCTS)** with UCT.

## What makes this demo different?
1. **Per-image decision making**  
2. **Explicit constraints** (size growth blocked by default, SSIM ≥ target)  
3. **Deployment realism** (limited action space + timeouts)

## Intended audience
- Engineers who want a deployable demo of search-based optimization
- Researchers looking for a compact prototype and ablation hooks
