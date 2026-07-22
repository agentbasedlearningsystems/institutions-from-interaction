"""Per-generation metrics logging for SISTER/GCoN agents.

One CSV row per CMA-ES generation (or per popsize steps in GCoN), per agent:
best/mean/std fitness, sigma, cache-hit/miss counts, AND mutual information
between displayed signs and chosen items across the logging window.

The MI measurement is the SISTER 2005 institutional-emergence test: rising
MI between agent signs and their behavior indicates that signs are acquiring
role-specific meaning.
"""
import csv
import math
import os
import re
from collections import defaultdict, deque

import numpy as np


def _item_category(item_str):
    """Keep distinct items distinct; collapse only trivial suffixes.

    'clusterer_stop' and 'clusterer_nltk_kmeans_20clusters' are distinct
    categories so the MI measurement can detect differentiation.
    """
    if not item_str:
        return "none"
    return item_str.strip()


def _mutual_information(pairs, n_sign_bins):
    """Empirical MI(sign, item) from a list of (sign_bin, item_category) pairs.

    Uses the plug-in estimator: I(X;Y) = sum p(x,y) log(p(x,y)/(p(x)p(y))).
    Returns 0 if fewer than 2 unique values in either variable.
    """
    if not pairs:
        return 0.0
    signs = [s for s, _ in pairs]
    items = [i for _, i in pairs]
    if len(set(signs)) < 2 or len(set(items)) < 2:
        return 0.0
    n = len(pairs)
    joint = defaultdict(int)
    px = defaultdict(int)
    py = defaultdict(int)
    for s, i in pairs:
        joint[(s, i)] += 1
        px[s] += 1
        py[i] += 1
    mi = 0.0
    for (s, i), c in joint.items():
        pxy = c / n
        p_x = px[s] / n
        p_y = py[i] / n
        if pxy > 0 and p_x > 0 and p_y > 0:
            mi += pxy * math.log2(pxy / (p_x * p_y))
    return mi


class _SignQuantizer:
    """Simple online vector quantizer for 8-D signs. Refit every N samples."""
    def __init__(self, k=8, refit_every=200):
        self.k = k
        self.refit_every = refit_every
        self.centers = None  # (k, d)
        self._buffer = []
        self._since_fit = 0

    def update(self, sign_vec):
        self._buffer.append(np.asarray(sign_vec, dtype=float))
        self._since_fit += 1
        if self.centers is None and len(self._buffer) >= self.k:
            self._fit()
        elif self._since_fit >= self.refit_every:
            self._fit()

    def _fit(self):
        X = np.stack(self._buffer[-self.refit_every * 2 :])
        # Lightweight kmeans: random init + 10 Lloyd iterations
        k = min(self.k, len(X))
        idx = np.random.choice(len(X), size=k, replace=False)
        centers = X[idx].copy()
        for _ in range(10):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            assign = d.argmin(1)
            for j in range(k):
                mask = assign == j
                if mask.any():
                    centers[j] = X[mask].mean(0)
        self.centers = centers
        self._since_fit = 0

    def quantize(self, sign_vec):
        if self.centers is None:
            return 0
        s = np.asarray(sign_vec, dtype=float)
        d = ((self.centers - s) ** 2).sum(-1)
        return int(d.argmin())


class MetricsLogger:
    COLUMNS = [
        "gen", "agent_id", "sim_time",
        "best_fit", "mean_fit", "std_fit", "sigma",
        "n_seen", "cache_hits", "cache_misses",
        "mi_sign_item", "n_pairs_in_mi",
    ]

    def __init__(self, output_path, sim, mi_window=200, sign_k=8):
        self.sim = sim
        os.makedirs(output_path, exist_ok=True)
        self.path = os.path.join(output_path, "metrics.csv")
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.COLUMNS)
        self.file.flush()
        self._gen_by_agent = defaultdict(int)

        # Rolling window of (sign_bin, item_category) pairs across all agents.
        # Sampled once per sim step via collect_pairs() called by SnetSim.
        self.mi_window = mi_window
        self.pairs = deque(maxlen=mi_window)
        self.quantizer = _SignQuantizer(k=sign_k, refit_every=mi_window)

    def collect_pairs(self):
        """Called once per sim step (by SnetSim.step). Snapshots the current
        blackboard into the sliding window."""
        for msg in self.sim.blackboard:
            sign = msg.get("sign") or []
            if not sign:
                continue
            self.quantizer.update(sign)
            s_bin = self.quantizer.quantize(sign)
            # take first trade's item as the agent's "behavior category"
            trades = msg.get("trades") or []
            item = trades[0].get("item") if trades else ""
            cat = _item_category(item)
            self.pairs.append((s_bin, cat))

    def log_generation(self, agent, results, sigma):
        agent_id = agent.unique_id
        self._gen_by_agent[agent_id] += 1
        gen = self._gen_by_agent[agent_id]

        valid = [r for r in results if r is not None]
        best = max(valid) if valid else ""
        mean = float(np.mean(valid)) if valid else ""
        std = float(np.std(valid)) if valid else ""

        cache = self.sim.cache
        hits = getattr(cache, "hit_count", "")
        misses = getattr(cache, "miss_count", "")

        pairs_snapshot = list(self.pairs)
        mi = _mutual_information(pairs_snapshot, n_sign_bins=self.quantizer.k)

        self.writer.writerow([
            gen, agent_id, self.sim.schedule.time,
            best, mean, std, sigma,
            len(valid), hits, misses,
            f"{mi:.4f}", len(pairs_snapshot),
        ])
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()
