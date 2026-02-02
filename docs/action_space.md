# Action Space Design

Search is only useful if the candidate set is:
- expressive enough to include good solutions
- small enough to evaluate under tight time budgets

## Why discrete qualities and scales?
Encoder behavior is not smooth; discrete choices stabilize results and reduce evaluation cost.

## Text-like vs photo profiles
Text-like images are sensitive to:
- edge blurring
- JPEG chroma subsampling
- resizing artifacts

Therefore text-like candidate generation is more conservative.
