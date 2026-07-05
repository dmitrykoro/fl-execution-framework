"""
Shared scoring primitives for LQR-Fed strategies.

Used by both `lqr_fed` (raw distances) and `lqr_fed_norm_dist` (mean-normalised
distances).  Keeping the math here lets the strategy class stay focused on the
Flower lifecycle and makes it easy to unit-test the QP independently.
"""

from typing import List

import numpy as np
from scipy.optimize import minimize


def flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    """Concatenate all parameter arrays into a single float64 vector."""
    return np.concatenate([w.flatten().astype(np.float64) for w in weights])


def compute_distances(
    flat_weights: List[np.ndarray],
    ref: np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute L2 distances from each client's flat weight vector to *ref*.

    Parameters
    ----------
    flat_weights : list of 1-D arrays, one per client.
    ref : 1-D reference vector (lagged global model or arithmetic mean).
    normalize : if True, divide every distance by the round-wise mean so that
        the QP sees *relative* deviation rather than absolute drift.  This
        makes q/r effective regardless of local epoch count or data
        heterogeneity, since all-clients-drifting-equally maps to all
        distances = 1.0 instead of all-distances-large.

    Returns
    -------
    1-D array of length N with non-negative distances (normalised if requested).
    """
    raw = np.array([float(np.linalg.norm(fw - ref)) for fw in flat_weights])
    if normalize:
        mean_d = raw.mean()
        if mean_d > 1e-12:
            return raw / mean_d
        # All clients have near-zero distance (e.g. warmup round 1); fall back
        # to uniform to avoid division-by-zero amplifying noise.
        return np.ones(len(raw))
    return raw


def solve_lqr_qp(distances: np.ndarray, q: float, r: float) -> np.ndarray:
    """
    Solve the LQR-inspired QP for per-client aggregation weights.

    Minimise   Σᵢ [ q·dᵢ²·wᵢ + r·(wᵢ − 1/N)² ]
    subject to Σ wᵢ = 1,   0 ≤ wᵢ ≤ 1

    The Hessian is 2r·I (positive definite), so the problem is strictly convex
    and SLSQP converges reliably in a small number of major iterations.

    Parameters
    ----------
    distances : 1-D array of per-client distances (raw or normalised).
    q : state cost — penalty on trusting high-distance clients.
    r : control cost — penalty on deviating from uniform weights.

    Returns
    -------
    1-D array of aggregation weights summing to 1, each in [0, 1].
    """
    N = len(distances)
    d_sq = distances ** 2
    uniform = 1.0 / N

    def objective(w: np.ndarray) -> float:
        return float(np.sum(q * d_sq * w) + r * np.sum((w - uniform) ** 2))

    def gradient(w: np.ndarray) -> np.ndarray:
        return q * d_sq + 2.0 * r * (w - uniform)

    result = minimize(
        objective,
        x0=np.full(N, uniform),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * N,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    weights = np.clip(result.x, 0.0, 1.0)
    total = weights.sum()
    return weights / total if total > 0 else np.full(N, uniform)


def weighted_aggregate(
    weight_lists: List[List[np.ndarray]],
    qp_weights: np.ndarray,
) -> List[np.ndarray]:
    """Layer-wise weighted average of client parameter lists using QP weights."""
    num_layers = len(weight_lists[0])
    aggregated = []
    for layer_idx in range(num_layers):
        stacked = np.stack([wl[layer_idx] for wl in weight_lists], axis=0)
        aggregated.append(np.tensordot(qp_weights, stacked, axes=(0, 0)))
    return aggregated
