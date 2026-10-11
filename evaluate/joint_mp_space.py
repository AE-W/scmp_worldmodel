"""Uniform baseline and fractional-policy MAC accounting."""
from evaluate.group_mp_policy import group_counts
from evaluate.search_recipe import DEFAULT_RECIPE


def uniform(names, budget=96):
    return dict(stoc_len_levels=[budget], level_fractions=[1.0],
                per_module_fractions={n: dict(level_fractions=[1.0], invert=False) for n in names})


def budget_cost(candidate, shapes, chunk=DEFAULT_RECIPE.chunk):
    """Shape-based linear budget for fixed fractional dispatch."""
    if type(chunk)!=int or chunk<1:
        raise ValueError('Input chunk must be a positive integer')
    total = units = 0
    for name,calls in shapes.items():
        d = candidate['per_module_fractions'][name]
        if 'group_thresholds' in d:
            raise ValueError('Threshold budgets require actual activations')
        for rows,width,outputs in calls:
            if width % chunk:
                raise ValueError(f'Incomplete input group in {name}')
            n = rows*(width//chunk)
            counts = group_counts(n,d['level_fractions'])
            total += outputs*sum(a*b for a,b in zip(candidate['stoc_len_levels'],counts))
            units += outputs*n
    if not units:
        raise ValueError('Empty budget profile')
    return total/units
