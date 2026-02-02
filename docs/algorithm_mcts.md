# MCTS Algorithm Details

This document describes the **two-level** MCTS used in MIO.

## Tree structure (2-level)
- Root
  - Codec node (one per codec)
    - Action nodes (one per concrete action in that codec)

This is intentionally shallow to keep evaluation stable on small servers.

## UCT selection
For a node i with parent p:

UCT(i) = mean_reward(i) + c * sqrt( ln(N_p + 1) / N_i )

Unvisited nodes are prioritized.

## Codec priors
Codec selection uses a lightweight prior bias to improve early exploration:
- sample unvisited codec nodes with prior weights
- later: UCT + small log-prior bonus

## Backprop
After evaluating an action, reward is propagated:
action → codec → root

Important: root visits are updated only via backprop (avoid double-counting).
