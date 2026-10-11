"""Ordered case-level execution with a persistent spawn-process worker pool."""
import atexit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from types import SimpleNamespace
import numpy as np

_EVALUATOR = None
_CASES = {}


def initialize_worker(arguments,splits):
    global _EVALUATOR
    from evaluate.joint_mp_search import Evaluator,make_cases
    cli = SimpleNamespace(**arguments)
    _EVALUATOR = Evaluator(cli)
    atexit.register(_EVALUATOR.close)
    calibration = make_cases(splits[0],_EVALUATOR.diffusion,_EVALUATOR.device,cli.seed,cli.draws,_EVALUATOR.expected_shape)
    _EVALUATOR.profile(calibration[0])
    _CASES.update({(c['key'],c['draw']):c for c in calibration})
    _EVALUATOR.parallel_cli = cli


def worker_case(identity):
    if identity not in _CASES:
        from evaluate.joint_mp_search import make_cases
        cli = _EVALUATOR.parallel_cli
        cases = make_cases([identity[0]],_EVALUATOR.diffusion,_EVALUATOR.device,
                          cli.seed,cli.confirmation_draws,_EVALUATOR.expected_shape)
        _CASES.update({(c['key'],c['draw']):c for c in cases})
    return _CASES[identity]


def execute_case(task):
    mode,identity,candidate,options = task
    case = worker_case(identity)
    if mode=='score':
        return _EVALUATOR.score(candidate,[case],**options)
    from evaluate.upstream_mp_measure import measure
    return measure(_EVALUATOR,[case],candidate,**options)


def merge_scores(results,weights,profile=False):
    total = sum(weights)
    if not results or len(results)!=len(weights) or total<=0:
        raise ValueError('Ordered case scores require matching positive weights')
    losses = [r['per_case'][0] for r in results]
    costs = [r['per_case_cycles'][0] for r in results]
    result = dict(mean=sum(x*w for x,w in zip(losses,weights))/total,per_case=losses,
                  avg_cycles=sum(x*w for x,w in zip(costs,weights))/total,per_case_cycles=costs)
    if profile:
        result['level_weights'] = sum(np.asarray(r['level_weights'])*w/total
                                      for r,w in zip(results,weights)).tolist()
        names = set(results[0]['metric_profile'])
        if any(set(r['metric_profile'])!=names for r in results):
            raise ValueError('Worker profile coverage differs')
        result['metric_profile'] = {name:sum(np.asarray(r['metric_profile'][name])*w/total
            for r,w in zip(results,weights)).tolist() for name in names}
    return result


class CaseWorkers:
    def __init__(self,cli,splits):
        self.pool = ProcessPoolExecutor(max_workers=cli.workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker,
            initargs=(vars(cli),splits))

    def score(self,candidate,cases,profile=False,profile_bins=257):
        tasks = [('score',(c['key'],c['draw']),candidate,
                  dict(profile=profile,profile_bins=profile_bins)) for c in cases]
        return merge_scores(list(self.pool.map(execute_case,tasks)),[c['weight'] for c in cases],profile)

    def measure(self,cases,candidate,rows):
        total_weight = sum(c['weight'] for c in cases)
        tasks = [('measure',(c['key'],c['draw']),candidate,
                  dict(rows=rows,total_weight=total_weight)) for c in cases]
        records,buckets = {},None
        for measured,mapping in self.pool.map(execute_case,tasks):
            if buckets is not None and mapping!=buckets:
                raise ValueError('Worker bucket coverage differs')
            buckets = mapping
            for name,values in measured.items(): records.setdefault(name,[]).extend(values)
        return records,buckets

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
