"""Exact fractional-dispatch cycle accounting."""


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
