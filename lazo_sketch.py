from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class ColumnSketch:
    """Container for LAZO-compatible sketches of a single column."""
    minhash_signature: np.ndarray
    cardinality: int
    column_name: str
    dataset_id: str


class ColumnSketcher:
    """
    Builds MinHash + approximate cardinality sketches for a pandas Series.
    - MinHash (K=128) gives a low-variance estimator of Jaccard similarity; signature size trades accuracy for memory.
    - HyperLogLog-style counter gives an approximate distinct count; large errors here will propagate to containment estimates.
    """
    def __init__(self, num_hashes: int = 128, hll_b: int = 10) -> None:
        """
        :param num_hashes: number of hash functions for MinHash; higher K lowers variance O(1/sqrt(K)).
        :param hll_b: HyperLogLog precision parameter; m=2^b registers. Larger b => lower error (~1.04/sqrt(m)) but more memory.
        """
        self.num_hashes = num_hashes
        self.hash_seeds = self._init_hash_seeds(num_hashes)
        self.hll_b = hll_b
        self.hll_m = 1 << hll_b  # number of registers

    def _init_hash_seeds(self, k: int) -> List[int]:
        rnd = random.Random(42)
        return [rnd.getrandbits(64) for _ in range(k)]

    def _hash_value(self, value: str, seed: int) -> int:
        data = f"{seed}:{value}".encode("utf-8")
        # Use a stable, fast 64-bit hash via SHA1 digest truncation
        return int(hashlib.sha1(data).hexdigest()[:16], 16)

    def _minhash_signature(self, values: Iterable[str]) -> np.ndarray:
        """Compute MinHash signature over cleaned values."""
        signature = np.full(self.num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
        for v in values:
            for i, seed in enumerate(self.hash_seeds):
                hv = self._hash_value(v, seed)
                if hv < signature[i]:
                    signature[i] = hv
        return signature

    def _hll_cardinality(self, values: Iterable[str]) -> int:
        """
        Very small HyperLogLog-style estimator.
        - Errors shrink with more registers (m), but m=1024 (b=10) keeps memory tiny.
        - Bias is ignored for simplicity; good enough for ranking joinability, not exact counts.
        """
        registers = [0] * self.hll_m
        for v in values:
            hv = self._hash_value(v, seed=0)
            idx = hv & (self.hll_m - 1)  # lower b bits
            w = hv >> self.hll_b
            leading = self._rho(w, 64 - self.hll_b)
            registers[idx] = max(registers[idx], leading)

        alpha_m = 0.7213 / (1 + 1.079 / self.hll_m)
        indicator = sum(2.0**-r for r in registers)
        raw_est = alpha_m * (self.hll_m**2) / indicator
        return max(1, int(raw_est))

    @staticmethod
    def _rho(w: int, max_bits: int) -> int:
        """Position of first 1-bit plus one; capped by max_bits."""
        if w == 0:
            return max_bits
        leading = (w.bit_length())
        return max_bits - leading + 1

    @staticmethod
    def _clean_series(series: pd.Series) -> Iterable[str]:
        """Normalize values to lowercase, stripped strings; drop empties and NaN."""
        for val in series.dropna():
            s = str(val).strip().lower()
            if s:
                yield s

    def build_sketch(self, series: pd.Series, column_name: str = "", dataset_id: str = "") -> ColumnSketch:
        """
        Build MinHash + HLL sketch for a column.
        Probabilistic trade-offs:
        - MinHash variance is driven by K; we fix K=128 to balance speed and accuracy.
        - HLL gives ~3% relative error with m=1024; good enough for containment thresholds, not exact distincts.
        """
        cleaned = list(self._clean_series(series))
        signature = self._minhash_signature(cleaned)
        card = self._hll_cardinality(cleaned)
        return ColumnSketch(
            minhash_signature=signature,
            cardinality=card,
            column_name=column_name or series.name,
            dataset_id=dataset_id,
        )
