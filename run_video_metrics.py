"""Paired decoded-video metrics for the frozen ten-case search policy."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from evaluate.search_io import digest, save
from evaluate.search_recipe import SearchRecipe, runtime_environment


def main():
    root = Path('results/search10-01').resolve()
    proposal = json.load(open(root / 'proposals/search_state.json'))
    rollout = json.load(open(root / 'rollout/state.json'))
    protocol = proposal['protocol']
    assert rollout['complete'] and rollout['selected']['candidate'] == json.load(open(root / 'rollout/policy.json'))
    recipe = SearchRecipe(**proposal['provenance']['recipe'])
    keys = json.load(open(protocol['calibration_keys']))
    assert len(keys) == 10 and len(set(keys)) == 10
    assert keys == json.load(open(root / 'rollout/protocol.json'))['search_keys']
    run = root / 'video-metrics'
    run.mkdir(mode=0o700, exist_ok=False)
    save(run / 'manifest.json', dict(keys=keys, seed=protocol['video_seed'],
         sampler=recipe.video_scheduler, steps=recipe.video_steps,
         source_policy_sha256=digest(root / 'rollout/policy.json'),
         source_state_sha256=digest(root / 'rollout/state.json'),
         config_sha256=digest(protocol['config']), scales_sha256=digest(protocol['scales'])))
    try:
        reference = json.load(open('results/run-01/paired_metrics_report.json'))
        if reference['keys'] != keys or reference['seed'] != protocol['video_seed']:
            raise ValueError('Historical uniform video metrics are not paired with this run')
        smoke = json.load(open(root / 'rollout/smoke.json'))
        if not smoke['uniform_rollout_identity']:
            raise ValueError('Uniform rollout parity not verified')
        scores = {'uniform': dict(mean=reference['means']['uniform'],
                                  per_case=reference['per_sample']['uniform'])}
        save(run / 'uniform_reference.json', dict(source='results/run-01/paired_metrics_report.json',
             source_sha256=digest('results/run-01/paired_metrics_report.json'),
             rollout_smoke_sha256=digest(root / 'rollout/smoke.json')))
        for arm, candidate in (('selected', rollout['selected']['candidate']),):
            config = run / (arm + '.json')
            save(config, candidate)
            jobs, logs = [], []
            for shard, gpu in enumerate((0, 2, 6, 7)):
                env = dict(runtime_environment(recipe), PYTHONPATH='.',
                    EVAL_PAIR_SEED=str(protocol['video_seed']),
                    SC_MP_GROUP_CHUNK_D=str(recipe.chunk),
                    SC_SMOOTH_SCALES=str(Path(protocol['scales']).resolve()),
                    SC_MP_PER_MODULE=str(config),
                    SC_MP_CONFIG=json.dumps({k: candidate[k] for k in ('stoc_len_levels', 'level_fractions')}),
                    CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2')
                log = open(run / f'{arm}_{shard}.log', 'w')
                logs.append(log)
                jobs.append(subprocess.Popen([sys.executable, 'evaluate/eval_local_n_samples.py',
                    '--config', protocol['config'], '--skip', protocol['skip'], '--tag', arm,
                    '--keys_file', protocol['calibration_keys'], '--num_samples', str(len(keys)),
                    '--shard', str(shard), '--num_shards', '4',
                    '--inference_steps', str(recipe.video_steps), '--scheduler', recipe.video_scheduler,
                    '--out_root', str(run / 'evaluation')], env=env, stdout=log, stderr=subprocess.STDOUT))
            codes = [job.wait() for job in jobs]
            for log in logs:
                log.close()
            if codes != [0] * 4:
                raise RuntimeError(f'{arm} worker exit codes: {codes}')
            met_dir = run / 'evaluation' / arm / 'metrics'
            rows = [json.load(open(met_dir / (key + '.json'))) for key in keys]
            if not all(row['key'] == key and row['generation_seed'] == protocol['video_seed']
                       and all(math.isfinite(row[m]) for m in ('psnr', 'ssim', 'latent_l2'))
                       for key, row in zip(keys, rows)):
                raise RuntimeError(f'{arm} incomplete or invalid metrics')
            scores[arm] = dict(mean={m: sum(row[m] for row in rows) / len(rows)
                                      for m in ('psnr', 'ssim', 'latent_l2')}, per_case=rows)
            save(run / 'progress.json', scores)
            print(arm, scores[arm]['mean'], flush=True)
        save(run / 'report.json', dict(scores=scores,
             wins={m: sum((a[m] > b[m]) if m != 'latent_l2' else (a[m] < b[m])
                          for a, b in zip(scores['selected']['per_case'], scores['uniform']['per_case']))
                   for m in ('psnr', 'ssim', 'latent_l2')},
             note='Search-set metrics; policy selected using these ten trajectories.'))
        save(run / 'status.json', dict(status='complete'))
    except BaseException as error:
        save(run / 'status.json', dict(status='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
