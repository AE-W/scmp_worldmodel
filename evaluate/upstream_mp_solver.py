"""IRASim adapter for upstream per-group monotone allocation.

Algorithm/reference: scmp_llm 59cb8f0, mp_per_row_chunk_calib5/8.py.
Independent implementation; DP is exact for a fixed lambda and binned proxy,
not a guarantee of the constrained/global task optimum.
"""
import numpy as np
from evaluate.search_recipe import DEFAULT_RECIPE

DENSE_LADDER = [8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128]
UPSTREAM_COMMIT = '59cb8f0f1164f638d931b472f3b64a8c5ab58ccb'


def normalized_metric(x, chunk=DEFAULT_RECIPE.chunk):
    """Normalize raw group absmax over the full call, BEFORE subsampling."""
    m = x.abs().reshape(x.shape[0], -1, chunk).amax(-1)
    return (m - m.min()) / (m.max() - m.min()).clamp_min(1e-8)


def staircase(errors, costs, lengths, price):
    """Minimum sum(error + price*MAC*length) under nondecreasing lengths."""
    errors, costs, lengths = map(np.asarray, (errors, costs, lengths))
    if errors.ndim != 2 or errors.shape != (len(costs), len(lengths)) or not len(costs):
        raise ValueError('Invalid DP inputs')
    if np.any(np.diff(lengths) <= 0) or np.any(costs < 0):
        raise ValueError('Lengths must ascend and costs must be nonnegative')
    previous = errors[0] + price * costs[0] * lengths
    parents = np.zeros(errors.shape, dtype=np.int32)
    for b in range(1, len(costs)):
        winner = 0
        best = np.empty(len(lengths))
        for rung in range(len(lengths)):
            if previous[rung] < previous[winner]:
                winner = rung
            parents[b, rung] = winner
            best[rung] = previous[winner]
        previous = best + errors[b] + price * costs[b] * lengths
    path = np.empty(len(costs), dtype=np.int32)
    path[-1] = np.argmin(previous)
    for b in range(len(costs)-1, 0, -1):
        path[b-1] = parents[b, path[b]]
    return path


def bin_records(records, bins=2048):
    """Upstream stable sort, equal-count bins and sampled expanded MACs.

    Ties may straddle bins, as upstream; deployment can consequently differ
    from proxy occupancy. Full-model measured budget repair handles this.
    """
    metric = np.concatenate([r['metric'] for r in records])
    if not len(metric) or bins < 1:
        raise ValueError('Empty measurements or invalid bin count')
    order = np.argsort(metric, kind='stable')
    edges = np.linspace(0,len(metric),min(bins,len(metric))+1).astype(np.int64)
    E = {}
    for currency in ('raw','fis'):
        error = np.concatenate([r[currency]-r[currency][:,[-1]] for r in records])[order]
        E[currency] = np.add.reduceat(error,edges[:-1],axis=0)
    mac = np.concatenate([np.full(len(r['metric']),r['mac_weight']*r['population']/r['sampled_rows'])
                          for r in records])[order]
    C = np.add.reduceat(mac,edges[:-1])
    return dict(errors=E,costs=C,edges=metric[order][edges[1:]-1])


def thresholds(path, edges, n_levels):
    result = []
    for rung in range(n_levels-1):
        low = np.flatnonzero(path <= rung)
        if not len(low):
            result.append(0.0)  # Upstream sentinel, including its m=0 behavior.
        elif low[-1] == len(path)-1:
            result.append(1.0)
        else:
            result.append(float(edges[low[-1]]))
    return result


def solve_global(binned, currency, budget, iterations=80, levels=None):
    lengths = np.asarray(DENSE_LADDER if levels is None else levels)
    units = sum(v['costs'].sum() for v in binned.values())
    if units <= 0 or not min(lengths) <= budget <= max(lengths):
        raise ValueError('Invalid global budget')

    def evaluate(price):
        paths = {k: staircase(v['errors'][currency], v['costs'], lengths, price)
                 for k, v in binned.items()}
        cost = sum(np.dot(binned[k]['costs'], lengths[p]) for k, p in paths.items()) / units
        return paths, float(cost)

    best, cost = evaluate(0)
    price = 0.0
    if cost > budget:
        low, high = 0.0, 1e-30
        best, cost = evaluate(high)
        while cost > budget and high < 1e40:
            high *= 10
            best, cost = evaluate(high)
        if cost > budget:
            raise RuntimeError('Could not bracket a feasible price')
        price = high
        for _ in range(iterations):
            mid = (low*high)**.5 if low else high / 1e3
            proposed, proposed_cost = evaluate(mid)
            if proposed_cost <= budget:
                best, cost, high, price = proposed, proposed_cost, mid, mid
            else:
                low = mid
            if low and high/low < 1+1e-7:
                break
    return {k: thresholds(p, binned[k]['edges'], len(lengths)) for k,p in best.items()}, dict(
        currency=currency, avg_cycles=cost, price=price,
        occupied_levels=sorted({int(lengths[i]) for p in best.values() for i in p}))


def policy(names_to_bucket, cuts, levels=None):
    # Keep the ascending ladder: torch.bucketize returns its direct index.
    levels = list(DENSE_LADDER if levels is None else levels)
    return dict(stoc_len_levels=levels, level_fractions=[1.0]+[0.0]*(len(levels)-1),
                dispatch='group_thresholds_v1', per_module_fractions={
                    n: dict(group_thresholds=cuts[b], invert=False) for n,b in names_to_bucket.items()})
