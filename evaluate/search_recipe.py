"""Shared runtime recipe and initial search-space validation."""
from dataclasses import asdict, dataclass
import os


@dataclass(frozen=True)
class SearchRecipe:
    precision: int = 8
    chunk: int = 128
    halve: bool = True
    owen_mode: str = 'bitrev'
    scramble_masks: int = 64
    layer_buckets: int = 4
    video_steps: int = 50
    video_scheduler: str = 'PNDM'

    def to_dict(self):
        return asdict(self)


DEFAULT_RECIPE = SearchRecipe()
KERNEL_MAX_LENGTH = 2 ** (DEFAULT_RECIPE.precision - 1)

RUNTIME_UNSET = (
    'SC_MP_CONFIG', 'SC_MP_PER_MODULE', 'SC_MP_GROUP_CHUNK_D',
    'SC_STEP_SCHEDULE', 'SC_MP_INVERT', 'SC_UNIFORM_STOC_LEN',
    'SC_MP_LEGACY_RAW_AMAX', 'SC_DISABLE_OWEN', 'SC_SCRAMBLE_RESCALE',
    'SC_FORCE_COMPACT', 'SC_FORCE_TABLE',
    'SC_MP_TRACE', 'SC_MP_TRACE_MODE', 'SC_MP_TRACE_MAX_SHAPES',
)


def runtime_settings(recipe):
    return dict(SC_PREC=str(recipe.precision), SC_HALVE=str(int(recipe.halve)),
        SC_MP_FIXED_PREC='1', SC_LINEAR_GRANULARITY='per_row',
        SC_OWEN_MODE=recipe.owen_mode, SC_SCRAMBLE_MASKS=str(recipe.scramble_masks),
        SC_HW_MAX_MASKS='64', NVIDIA_TF32_OVERRIDE='0',
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',
        SC_CUM_INDICATOR_CACHE='1', SC_ENABLE_TABLE_CACHE_MAX='32')


def runtime_environment(recipe, base=None):
    env = dict(os.environ if base is None else base)
    for knob in RUNTIME_UNSET:
        env.pop(knob, None)
    env.update(runtime_settings(recipe))
    return env


def configure_float_precision(torch):
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_ladder(values, minimum, maximum, budget):
    if any(type(v)!=int for v in (minimum,maximum,budget)):
        raise ValueError('Lengths and the uniform reference budget must be integers')
    if not 1 <= minimum <= budget <= maximum <= KERNEL_MAX_LENGTH:
        raise ValueError('Require kernel-compatible bounds containing the integer budget')
    if not values or values != sorted(set(values)) or any(type(v)!=int or not 1<=v<=KERNEL_MAX_LENGTH for v in values):
        raise ValueError('Initial ladder must contain ascending unique integer lengths')
    if any(not minimum<=v<=maximum for v in values):
        raise ValueError('Every supplied ladder length must lie within the configured bounds')
    if not values[0]<=budget<=values[-1]:
        raise ValueError('Supplied ladder must span the target budget')
    return list(values)


def default_ladder(values, minimum, maximum, budget):
    """Clip the default grid, adding a bound only when required to span the budget."""
    levels = [v for v in values if minimum<=v<=maximum]
    if not levels or min(levels)>budget:
        levels.append(minimum)
    if not levels or max(levels)<budget:
        levels.append(maximum)
    return resolve_ladder(sorted(set(levels)),minimum,maximum,budget)


def uniform_variants(names, budget, ladder, minimum=1, maximum=KERNEL_MAX_LENGTH):
    levels = sorted(set([minimum,budget,maximum]),reverse=True)
    fractions = [float(v==budget) for v in levels]
    padded = dict(stoc_len_levels=levels,level_fractions=fractions,
        per_module_fractions={n:dict(level_fractions=fractions,invert=False) for n in names})
    levels = sorted(set(ladder+[budget]))
    cuts = [-1.0 if v<budget else 1.0 for v in levels[:-1]]
    threshold = dict(stoc_len_levels=levels,level_fractions=[1.0]+[0.0]*(len(levels)-1),
        dispatch='group_thresholds_v1',per_module_fractions={n:dict(group_thresholds=cuts,invert=False) for n in names})
    return padded,threshold
