from __future__ import annotations

from typing import Tuple

import numpy as np

from lazo_sketch import ColumnSketch


def estimate_js_from_minhash(sig_x: np.ndarray, sig_y: np.ndarray) -> float:
    """Standard MinHash Jaccard estimator: fraction of matching positions."""
    if sig_x.shape != sig_y.shape:
        raise ValueError("Signatures must have the same length")
    if len(sig_x) == 0:
        return 0.0
    return float(np.mean(sig_x == sig_y))


def estimate_alpha(js_hat: float, card_x: int, card_y: int) -> float:
    """
    Cardinality-coupled alpha estimate from LAZO.
    Alpha represents the estimated overlap size adjusted for asymmetric cardinalities.
    """
    mn = min(card_x, card_y)
    mx = max(card_x, card_y)
    return (mn - js_hat * mx) / (1.0 + js_hat)


def estimate_js_jc_lazo(sketch_x: ColumnSketch, sketch_y: ColumnSketch) -> Tuple[float, float, float]:
    """
    Compute coupled Jaccard (JS) and containment (JC) using LAZO with error correction.

    - JS and JC are coupled through alpha: alpha captures the overlap implied by JS and the two cardinalities.
    - Alpha is needed because MinHash alone gives symmetric JS, but containment needs directionality and sizes.
    - Error Correction Heuristic (ECH): if estimated containment exceeds its theoretical max (min/size),
      we treat MinHash JS as overestimated, adjust alpha so containment hits the bound, and recompute JS for consistency.
    """
    card_x = sketch_x.cardinality
    card_y = sketch_y.cardinality

    js_hat = estimate_js_from_minhash(sketch_x.minhash_signature, sketch_y.minhash_signature)
    alpha_hat = estimate_alpha(js_hat, card_x, card_y)

    mn = min(card_x, card_y)
    jc_x = (mn - alpha_hat) / card_x if card_x else 0.0
    jc_y = (mn - alpha_hat) / card_y if card_y else 0.0

    # Upper bounds
    jc_x_max = mn / card_x if card_x else 0.0
    jc_y_max = mn / card_y if card_y else 0.0

    corrected = False
    # If JC exceeds its max, reduce alpha so JC hits its bound, then recompute JS from the new alpha.
    if jc_x > jc_x_max + 1e-12:
        alpha_hat = mn - jc_x_max * card_x
        corrected = True
    if jc_y > jc_y_max + 1e-12:
        alpha_hat = mn - jc_y_max * card_y
        corrected = True

    if corrected:
        # Recompute JS from corrected alpha: invert the alpha formula.
        mx = max(card_x, card_y)
        js_corrected = (mn - alpha_hat) / (mx + alpha_hat)
        js_hat = max(0.0, min(1.0, js_corrected))

        # Recompute containments from corrected alpha.
        jc_x = (mn - alpha_hat) / card_x if card_x else 0.0
        jc_y = (mn - alpha_hat) / card_y if card_y else 0.0

    return float(js_hat), float(jc_x), float(jc_y)
