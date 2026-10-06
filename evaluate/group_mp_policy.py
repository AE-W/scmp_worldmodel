"""Bounded deployable policy space and exact quantile cycle accounting."""


def policies(levels):
    out = []
    for i in range(len(levels)):
        f = [0.0] * len(levels)
        f[i] = 1.0
        out.append(dict(level_fractions=f, invert=False))
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            for fraction in (.25, .5, .75):
                f = [0.0] * len(levels)
                f[i], f[j] = fraction, 1 - fraction
                for invert in (False, True):
                    out.append(dict(level_fractions=f, invert=invert))
    return out


def mean_cycles(n_groups, levels, fractions):
    if n_groups <= 0:
        raise ValueError("No groups to dispatch")
    offset = total = 0
    for i, (length, fraction) in enumerate(zip(levels, fractions)):
        count = round(fraction * n_groups) if i < len(levels) - 1 else n_groups - offset
        count = max(0, min(count, n_groups - offset))
        total += length * count
        offset += count
    if offset != n_groups:
        raise ValueError("Incomplete group accounting")
    return total / n_groups
