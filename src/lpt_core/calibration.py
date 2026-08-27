from __future__ import annotations

import numpy as np


def pairwise_gap_epsilon(test_logits: np.ndarray, reference_logits: np.ndarray) -> float:
    """Empirical pairwise-gap error bound.

    If e_v = test_logit_v - reference_logit_v, then
    max_{u,v} |(e_v - e_u)| is max(e) - min(e). The final paper uses epsilon
    directly as this pairwise-gap error bound.
    """

    err = np.asarray(test_logits, dtype=np.float64) - np.asarray(reference_logits, dtype=np.float64)
    if err.shape[-1] < 1:
        raise ValueError("logit arrays must have a vocabulary dimension")
    ranges = err.max(axis=-1) - err.min(axis=-1)
    return float(np.max(ranges))

