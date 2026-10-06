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


def group_counts(n_groups, fractions):
    if n_groups <= 0:
        raise ValueError("No groups to dispatch")
    offset = 0
    counts = []
    for i, fraction in enumerate(fractions):
        count = round(fraction * n_groups) if i < len(fractions) - 1 else n_groups - offset
        count = max(0, min(count, n_groups - offset))
        counts.append(count)
        offset += count
    if offset != n_groups:
        raise ValueError("Incomplete group accounting")
    return counts


def mean_cycles(n_groups, levels, fractions):
    return sum(length * count for length, count in zip(levels, group_counts(n_groups, fractions))) / n_groups
