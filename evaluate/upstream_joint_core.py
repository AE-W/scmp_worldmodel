"""Pure CPU routines ported from scmp_llm at 59cb8f0.

Sources: benchmark/ppl/mp_ladder_refine.py and mp_joint_refine.py.
Integer routines retain upstream numerics. Boundary comparison supports
the ascending per-row-chunk dispatch convention through inclusive_high.
"""
from typing import Iterable

def weighted_cost(
    levels: Iterable[int],
    level_weights: Iterable[float],
    protected_weight: float = 0.0,
    protected_stoc_len: int = 0,
) -> float:
    """MAC-weighted stream length for adaptive + fixed protected work."""
    return (
        sum(float(sl) * float(w) for sl, w in zip(levels, level_weights))
        + float(protected_weight) * float(protected_stoc_len)
    )

def propose_floor_exchange(
    levels: Iterable[int],
    level_weights: Iterable[float],
    *,
    target_cost: float,
    protected_weight: float = 0.0,
    protected_stoc_len: int = 0,
    floor_step: int = 4,
    donor_step: int = 1,
    min_gap: int = 1,
    extra_cost: float = 0.0,
) -> dict:
    """Raise the floor, then greedily project higher rungs back to budget.

    Raising the floor can collide with the rung above it after several rounds.
    In that case the receiver-side increase propagates upward just enough to
    keep the ladder strictly descending. Donor reductions then spend from the
    higher rungs while preserving the same ordering constraint.

    Every intermediate donor state is considered and the state closest to the
    target is returned. Occupancy is frozen for the projection; the candidate's
    actual trace is measured after evaluation and is the acceptance authority.
    """
    old = [int(x) for x in levels]
    weights = [float(x) for x in level_weights]
    if len(old) < 2 or len(old) != len(weights):
        raise ValueError("levels and level_weights must have equal length >= 2")
    if floor_step <= 0 or donor_step <= 0 or min_gap <= 0:
        raise ValueError("floor_step, donor_step, and min_gap must be positive")
    if any(old[i] <= old[i + 1] for i in range(len(old) - 1)):
        raise ValueError(f"levels must be strictly descending, got {old}")

    candidate = old.copy()
    candidate[-1] += int(floor_step)
    # Receiver propagation: preserve the class count/threshold topology when a
    # raised floor catches the rung immediately above it.
    for i in range(len(candidate) - 2, -1, -1):
        candidate[i] = max(candidate[i], candidate[i + 1] + int(min_gap))

    def cost(xs: list[int]) -> float:
        # extra_cost: constant non-ladder work (escape-gate MAC share at its
        # fixed escape length) — priced like the protected term.
        return weighted_cost(
            xs, weights, protected_weight, protected_stoc_len) + float(extra_cost)

    receiver_levels = candidate.copy()
    receiver_cost = cost(candidate)
    best = candidate.copy()
    best_cost = receiver_cost
    path: list[dict] = []

    # At most a few hundred one-cycle moves for current ladders. Zero-occupancy
    # rungs are allowed to move because doing so can unlock the donor above.
    for _ in range(10000):
        moves = []
        for i in range(len(candidate) - 1):
            lowered = candidate[i] - int(donor_step)
            if lowered < candidate[i + 1] + int(min_gap):
                continue
            trial = candidate.copy()
            trial[i] = lowered
            trial_cost = cost(trial)
            moves.append((trial_cost, i, trial))
        if not moves:
            break
        # Prefer not to undershoot and spend from the highest rung first, which
        # is the user's proposed floor-for-ceiling exchange. If every available
        # one-cycle move would cross the target, take the smallest overshoot.
        feasible = [m for m in moves if m[0] >= target_cost]
        if feasible:
            trial_cost, donor_idx, trial = min(feasible, key=lambda m: m[1])
        else:
            trial_cost, donor_idx, trial = min(
                moves, key=lambda m: (abs(m[0] - target_cost), m[1]))
        candidate = trial
        path.append({
            "donor_index": int(donor_idx),
            "donor_level": int(candidate[donor_idx]),
            "predicted_cost": float(trial_cost),
        })
        if abs(trial_cost - target_cost) < abs(best_cost - target_cost):
            best = candidate.copy()
            best_cost = trial_cost
        if trial_cost <= target_cost:
            break

    if best == old:
        raise ValueError("floor exchange produced no ladder change")
    return {
        "old_levels": old,
        "receiver_levels": receiver_levels,
        "levels": best,
        "target_cost": float(target_cost),
        "receiver_cost": float(receiver_cost),
        "predicted_cost": float(best_cost),
        "predicted_budget_error": float(best_cost - target_cost),
        "floor_step": int(floor_step),
        "donor_step": int(donor_step),
        "min_gap": int(min_gap),
        "donor_moves": path,
    }

def project_levels_to_budget(
    levels: Iterable[int],
    level_weights: Iterable[float],
    *,
    target_cost: float,
    protected_weight: float = 0.0,
    protected_stoc_len: int = 0,
    min_gap: int = 1,
    min_level: int = 1,
    max_level: int = 128,
) -> dict:
    """Project an ordered integer ladder to a target measured occupancy.

    Under-budget ladders spend on the lowest rung first (propagating upward
    only when ordering requires it).  Over-budget ladders remove cycles from
    the highest rung first.  Every one-cycle state is considered and the
    closest state is returned.
    """
    current = [int(x) for x in levels]
    weights = [float(x) for x in level_weights]
    if len(current) < 2 or len(current) != len(weights):
        raise ValueError("levels and weights must have equal length >= 2")
    if any(current[i] <= current[i + 1]
           for i in range(len(current) - 1)):
        raise ValueError(f"levels must be strictly descending: {current}")
    if not (0 < min_level <= max_level):
        raise ValueError("invalid level bounds")

    def cost(xs: list[int]) -> float:
        return weighted_cost(
            xs, weights, protected_weight, protected_stoc_len)

    start = current.copy()
    start_cost = cost(current)
    best = current.copy()
    best_cost = start_cost
    path: list[dict] = []

    for _ in range(20000):
        current_cost = cost(current)
        under = current_cost < target_cost
        trials = []
        if under:
            # Prefer floor uplift, then progressively higher receivers.
            for receiver in range(len(current) - 1, -1, -1):
                trial = current.copy()
                trial[receiver] += 1
                for i in range(receiver - 1, -1, -1):
                    trial[i] = max(trial[i], trial[i + 1] + min_gap)
                if trial[0] > max_level:
                    continue
                trials.append((receiver, trial, cost(trial)))
        else:
            # Prefer taking cycles from the high rungs.  Do not propagate a
            # decrement downward: another independently feasible donor is
            # cleaner and preserves the floor whenever possible.
            for donor in range(0, len(current)):
                lowered = current[donor] - 1
                lower_bound = (current[donor + 1] + min_gap
                               if donor + 1 < len(current) else min_level)
                if lowered < lower_bound:
                    continue
                trial = current.copy()
                trial[donor] = lowered
                trials.append((donor, trial, cost(trial)))
        if not trials:
            break
        # Direction preference is already encoded in the index ordering.
        idx, trial, trial_cost = min(
            trials, key=lambda x: (abs(x[2] - target_cost),
                                   -x[0] if under else x[0]))
        if trial == current:
            break
        current = trial
        path.append({
            "direction": "up" if under else "down",
            "index": int(idx),
            "levels": current.copy(),
            "predicted_cost": float(trial_cost),
        })
        if abs(trial_cost - target_cost) < abs(best_cost - target_cost):
            best = current.copy()
            best_cost = trial_cost
        # Once the next directed step crossed the target and did not improve
        # the closest state, further steps in that direction cannot help.
        if ((under and trial_cost >= target_cost)
                or (not under and trial_cost <= target_cost)):
            break

    return {
        "type": "integer_budget_projection",
        "old_levels": start,
        "levels": best,
        "target_cost": float(target_cost),
        "start_cost": float(start_cost),
        "predicted_cost": float(best_cost),
        "predicted_budget_error": float(best_cost - target_cost),
        "min_gap": int(min_gap),
        "min_level": int(min_level),
        "max_level": int(max_level),
        "moves": path,
    }

def _shift_one_boundary(
    thresholds: list[float],
    hist: list[float],
    values: list[float],
    *,
    boundary: int,
    mass_delta: float,
    inclusive_high: bool = True,
) -> tuple[list[float], dict]:
    n_boundaries = len(thresholds)
    idx = boundary if boundary >= 0 else n_boundaries + boundary
    if not 0 <= idx < n_boundaries:
        raise ValueError(f"invalid boundary {boundary} for {thresholds}")
    total = sum(hist)
    if total <= 0.0:
        return thresholds.copy(), {"changed": False, "reason": "empty_hist"}
    old = float(thresholds[idx])
    def high_mass(threshold):
        return sum(w for v,w in zip(values,hist)
                   if (v >= threshold if inclusive_high else v > threshold))
    old_high = high_mass(old)
    target_high = min(max(old_high + mass_delta * total, 0.0), total)
    lower = float(thresholds[idx + 1]) if idx + 1 < n_boundaries else 0.0
    upper = float(thresholds[idx - 1]) if idx > 0 else 1.0
    candidates = sorted({lower, upper, old, *(
        v for v in values if lower <= v <= upper
    )})
    chosen = min(
        candidates,
        key=lambda t: (
            abs(high_mass(t) - target_high),
            abs(t - old),
        ),
    )
    out = thresholds.copy()
    out[idx] = float(chosen)
    if any(out[i] < out[i + 1] - 1e-12
           for i in range(len(out) - 1)):
        raise AssertionError(f"threshold shift broke ordering: {out}")
    new_high = high_mass(chosen)
    return out, {
        "changed": abs(chosen - old) > 1e-12,
        "boundary": int(idx),
        "old_threshold": old,
        "new_threshold": float(chosen),
        "requested_mass_delta": float(mass_delta),
        "realized_mass_delta": float((new_high - old_high) / total),
    }
