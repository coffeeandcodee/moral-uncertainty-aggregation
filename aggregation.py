"""
Aggregation rules for combining framework scores into a single decision.

Each rule takes per-framework scores for one response and returns a scalar.
The four rules below are the ones compared in the experiment:

- Expected choiceworthiness (EC): weighted sum across frameworks.
- Maximin: the lowest framework score (risk-averse).
- Nash parliament: product of framework scores.
- Baseline: utilitarian score alone (ignores the other frameworks).
"""

from __future__ import annotations

from typing import Iterable, Mapping

DEFAULT_WEIGHTS = (0.4, 0.35, 0.25)  # utilitarian, deontological, ubuntu


def expected_choiceworthiness(
    u: float, d: float, ub: float, weights: Iterable[float] = DEFAULT_WEIGHTS
) -> float:
    wu, wd, wub = weights
    return round(wu * u + wd * d + wub * ub, 2)


def maximin(u: float, d: float, ub: float) -> float:
    return min(u, d, ub)


def nash(u: float, d: float, ub: float) -> float:
    return u * d * ub


def baseline(u: float, d: float, ub: float) -> float:
    return u


METHODS = {
    "ec": expected_choiceworthiness,
    "maximin": maximin,
    "nash": nash,
    "baseline": baseline,
}


def compute_row(score: Mapping[str, float], weights=DEFAULT_WEIGHTS) -> dict:
    """Augment a score row with all four aggregation values."""
    u, d, ub = score["utilitarian"], score["deontological"], score["ubuntu"]
    return {
        "id": score["id"],
        "utilitarian": u,
        "deontological": d,
        "ubuntu": ub,
        "ec": expected_choiceworthiness(u, d, ub, weights),
        "maximin": maximin(u, d, ub),
        "nash": nash(u, d, ub),
        "baseline": baseline(u, d, ub),
    }


def pick_winners(rows: list[dict]) -> dict:
    """Return the best-scoring row id and value for each aggregation method."""
    winners: dict[str, dict] = {}
    for method in METHODS:
        best = max(rows, key=lambda r: r[method])
        winners[method] = {"id": best["id"], "value": best[method]}
    return winners
