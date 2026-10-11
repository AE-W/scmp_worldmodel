"""IRASim adapter for upstream V11 joint refinement at 59cb8f0.

Same integer projection, floor exchange, stage order and acceptance rule.
Only operator names and ascending per-row-chunk dispatch are adapted.
"""
import copy
import hashlib
import json
import math
import numpy as np
from evaluate.upstream_joint_core import project_levels_to_budget, propose_floor_exchange, _shift_one_boundary
from evaluate.search_recipe import KERNEL_MAX_LENGTH

DEFAULT_BUDGET_TOL = .35

def within_budget(actual,target=96,tolerance=DEFAULT_BUDGET_TOL):
    if not math.isfinite(target) or target<=0 or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('Invalid target budget or absolute tolerance')
    return math.isfinite(actual) and actual>0 and abs(actual-target)<=tolerance+1e-12

def key(candidate):
    return hashlib.sha256(json.dumps(candidate,sort_keys=True).encode()).hexdigest()

def threshold_uniform(names,budget=96):
    return dict(stoc_len_levels=[budget],level_fractions=[1.0],
                dispatch='group_thresholds_v1',per_module_fractions={
                    n:dict(group_thresholds=[],invert=False) for n in names})

def canonical(candidate,min_level=1,max_level=KERNEL_MAX_LENGTH):
    """Validate, retaining zero-occupancy rungs and the parent's topology."""
    allowed = {'stoc_len_levels','level_fractions','dispatch','per_module_fractions'}
    if not isinstance(candidate,dict) or set(candidate)-allowed:
        raise ValueError('Group policy supports only lengths, dispatch, fractions and module thresholds')
    if candidate.get('dispatch','group_thresholds_v1')!='group_thresholds_v1':
        raise ValueError('Policy requires group_thresholds_v1 dispatch')
    if not isinstance(candidate.get('per_module_fractions'),dict) or not candidate['per_module_fractions']:
        raise ValueError('Policy requires nonempty module thresholds')
    c = copy.deepcopy(candidate)
    levels = c.get('stoc_len_levels')
    if not 1<=min_level<=max_level<=KERNEL_MAX_LENGTH:
        raise ValueError('Invalid kernel-compatible length bounds')
    if (not isinstance(levels,list) or not levels or any(type(L)!=int for L in levels)
            or levels!=sorted(set(levels)) or any(not min_level<=L<=max_level for L in levels)):
        raise ValueError('Invalid integer refinement ladder')
    for d in c['per_module_fractions'].values():
        if not isinstance(d,dict) or set(d)-{'group_thresholds','invert'}:
            raise ValueError('Module policy supports only group_thresholds and invert')
        cuts = d.get('group_thresholds')
        if not isinstance(cuts,list):
            raise ValueError('Module policy requires a list of group thresholds')
        if (any(type(t) not in (int,float) for t in cuts) or len(cuts)!=len(levels)-1
                or cuts!=sorted(cuts) or not all(math.isfinite(t) and -1<=t<=1 for t in cuts)):
            raise ValueError('Invalid refinement cuts')
        if d.get('invert',False):
            raise ValueError('Refinement requires ascending metric dispatch')
    c['level_fractions'] = [1.0]+[0.0]*(len(levels)-1)
    return c

def with_levels(candidate, descending,min_level=1,max_level=KERNEL_MAX_LENGTH):
    out = copy.deepcopy(candidate)
    out['stoc_len_levels'] = list(reversed(descending))
    return canonical(out,min_level,max_level)

def project(candidate, weights, budget, min_level=1, max_level=128):
    action = project_levels_to_budget(candidate['stoc_len_levels'][::-1],weights[::-1],
        target_cost=budget,min_level=min_level,max_level=max_level)
    return with_levels(candidate,action['levels'],min_level,max_level),action

def bucket_names(candidate,buckets):
    if set(candidate['per_module_fractions'])!=set(buckets):
        raise ValueError('Policy and bucket module coverage differ')
    result = {}
    for name,b in buckets.items():
        result.setdefault(b,[]).append(name)
    for b,names in result.items():
        cuts = [candidate['per_module_fractions'][n]['group_thresholds'] for n in names]
        if any(v!=cuts[0] for v in cuts[1:]):
            raise ValueError(f'Inconsistent shared thresholds in {b}')
    return result

def profile_weights(candidate, profile, buckets):
    """Frozen-profile estimate; measured runtime occupancy accepts a move."""
    raw = np.zeros(len(candidate['stoc_len_levels']))
    representatives = {b:names[0] for b,names in bucket_names(candidate,buckets).items()}
    if set(profile)!=set(representatives):
        raise ValueError('Metric profile and policy bucket coverage differ')
    for b,h in profile.items():
        h = np.asarray(h,dtype=float)
        values = np.arange(len(h))/(len(h)-1)
        cuts = candidate['per_module_fractions'][representatives[b]]['group_thresholds']
        idx = np.searchsorted(cuts,values,side='left')
        raw += np.bincount(idx,weights=h,minlength=len(raw))
    if raw.sum()<=0:
        raise ValueError('Empty metric profile')
    return (raw/raw.sum()).tolist()

# Active V11 operator groups under the fixed-attention IRASim recipe.
THRESHOLD_SPECS = (
    ('promote_mlp_floor', ('mlp_fc1','mlp_fc2')),
    ('promote_attn_linear_floor', ('qkv','proj')),
)

def threshold_proposals(candidate, profile, buckets, mass_step=.02):
    if len(candidate['stoc_len_levels'])<2:
        return []
    representatives = bucket_names(candidate,buckets)
    result = []
    for label,ops in THRESHOLD_SPECS:
        out = copy.deepcopy(candidate)
        changed = []
        for b,names in representatives.items():
            if b.split(':')[0] not in ops:
                continue
            hist = list(profile[b])
            values = [i/(len(hist)-1) for i in range(len(hist))]
            # Descending V11 floor boundary = last; ascending PRC = first.
            descending = out['per_module_fractions'][names[0]]['group_thresholds'][::-1]
            shifted,action = _shift_one_boundary(descending,hist,values,boundary=-1,
                mass_delta=mass_step,inclusive_high=False)
            for n in names:
                out['per_module_fractions'][n]['group_thresholds'] = shifted[::-1]
            changed.append(dict(bucket=b,**action))
        if any(a['changed'] for a in changed):
            result.append((label,canonical(out),changed))
    return result

def refine(initial, initial_score, score, buckets, budget=96, ladder_rounds=2,
           threshold_rounds=1, floor_step=4, mass_step=.02, budget_corrections=2,
           min_level=1,max_level=128,min_improvement=0.0,
           on_trial=lambda *args:None,budget_tol=DEFAULT_BUDGET_TOL):
    """V11 budget repair -> ladder -> threshold, all accepted by actual trace."""
    current,current_score = canonical(initial,min_level,max_level),initial_score
    bucket_names(current,buckets)
    history = []

    def evaluate(candidate,stage,iteration,action):
        result = score(candidate)
        if not all(math.isfinite(result[f]) for f in ('mean','avg_cycles')):
            raise ValueError('Nonfinite refinement result')
        record = dict(stage=stage,round=iteration,candidate_key=key(candidate),action=action,
            loss=result['mean'],avg_cycles=result['avg_cycles'],
            budget_error=result['avg_cycles']-budget,accepted=False,
            feasible=within_budget(result['avg_cycles'],budget,budget_tol))
        history.append(record)
        on_trial(candidate,result,stage,iteration)
        return result,record

    # Upstream includes the initial attempt plus budget_corrections retries.
    for correction in range(budget_corrections+1):
        if within_budget(current_score['avg_cycles'],budget,budget_tol):
            break
        if len(current['stoc_len_levels'])<2:
            break
        candidate,action = project(current,current_score['level_weights'],budget,min_level,max_level)
        if key(candidate)==key(current):
            break
        result,record = evaluate(candidate,'budget_projection',correction,action)
        if abs(result['avg_cycles']-budget)>=abs(current_score['avg_cycles']-budget):
            break
        record['accepted'] = True  # Budget projection acceptance uses actual budget error.
        current,current_score = candidate,result
    if not within_budget(current_score['avg_cycles'],budget,budget_tol):
        return current,current_score,history

    for iteration in range(1,ladder_rounds+1):
        if len(current['stoc_len_levels'])<2:
            break
        try:
            action = propose_floor_exchange(current['stoc_len_levels'][::-1],
                current_score['level_weights'][::-1],target_cost=budget,floor_step=floor_step)
            candidate = with_levels(current,action['levels'],min_level,max_level)
        except ValueError as exc:
            history.append(dict(stage='ladder',round=iteration,accepted=False,reason=str(exc)))
            break
        result,record = evaluate(candidate,'ladder',iteration,action)
        if not record['feasible'] or current_score['mean']-result['mean']<=min_improvement:
            break
        record['accepted'] = True
        current,current_score = candidate,result

    for iteration in range(1,threshold_rounds+1):
        eligible = []
        for label,moved,moves in threshold_proposals(current,current_score['metric_profile'],buckets,mass_step):
            weights = profile_weights(moved,current_score['metric_profile'],buckets)
            candidate,projection = project(moved,weights,budget,min_level,max_level)
            result,record = evaluate(candidate,'threshold',iteration,
                dict(candidate=label,threshold_moves=moves,level_projection=projection))
            if record['feasible'] and current_score['mean']-result['mean']>min_improvement:
                eligible.append((candidate,result,record))
        if not eligible:
            break
        current,current_score,record = min(eligible,key=lambda x:x[1]['mean'])
        record['accepted'] = True
    return current,current_score,history

def select_on_search(records):
    """Select the feasible calibration incumbent with minimum task loss."""
    eligible = [r for r in records if r['feasible']]
    if not eligible:
        raise ValueError('No iso-budget search incumbent')
    return min(eligible,key=lambda r:(r['calibration']['mean'],len(r['candidate']['stoc_len_levels'])))
