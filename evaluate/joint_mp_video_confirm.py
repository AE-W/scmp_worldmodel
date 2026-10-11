"""Paired video confirmation of the frozen calibration policy."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from evaluate.search_io import digest,save,file_manifest,runtime_sources,software_versions,hardware_versions,prepare_video_run,complete_video_run
from evaluate.search_recipe import SearchRecipe,runtime_environment,runtime_settings
from evaluate.joint_mp_space import uniform
from evaluate.eval_local_n_samples import GT_LATENT_DIR,GT_VIDEO_DIR,ANNOT_DIR,build_args
from evaluate.task_loss_refine import canonical
from evaluate.confirmation_status import begin_confirmation,confirmation_budget_status,finish_confirmation
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    cli = p.parse_args()
    root = Path(cli.out).resolve()
    state = json.load(open(root/'search_state.json'))
    protocol,provenance = state['protocol'],state['provenance']
    if not state.get('complete'):
        raise ValueError('Video confirmation requires completed search')
    begin_confirmation(root)
    recipe = SearchRecipe(**provenance['recipe'])
    if provenance['runtime_settings']!=runtime_settings(recipe):
        raise ValueError('Numerical runtime recipe changed after search')
    args = build_args(protocol['config'],recipe.video_steps)
    checks = (
        (protocol['config'],provenance['config_sha256']),
        (args.evaluate_checkpoint,provenance['checkpoint_sha256']),
        (protocol['scales'],state['scales_sha256']),
        (protocol['holdout_keys'],provenance['split_sha256'][1]),
    )
    for path,expected in checks:
        if digest(path)!=expected:
            raise ValueError(f'Confirmation input changed: {path}')
    kernel = subprocess.check_output(['git','-C','kernels','rev-parse','HEAD'],text=True).strip()
    if kernel!=provenance['kernel_commit'] or runtime_sources()!=provenance['runtime_sources']:
        raise ValueError('Runtime sources or kernel changed after search')
    if software_versions()!=provenance['software_versions']:
        raise ValueError('Runtime software changed after search')
    if hardware_versions(torch)!=provenance['hardware_versions']:
        raise ValueError('Evaluation GPU, CUDA runtime or driver changed after search')
    keys_file = Path(protocol['holdout_keys']).resolve()
    keys = json.load(open(keys_file))
    proposal = json.load(open(root/'search_policy.json'))
    if canonical(proposal,protocol['min_level'],protocol['max_level'])!=proposal:
        raise ValueError('Frozen policy must use the canonical group policy schema')
    noise = json.load(open(root/'noise_validation.json'))
    if proposal!=noise['selected']['candidate'] or proposal!=state['trials'][state['selected_index']]['candidate']:
        raise ValueError('Confirmation policy differs from the frozen calibration policy')
    confirmed = {c['key']:c for c in noise['manifest']}
    if set(confirmed)!=set(keys):
        raise ValueError('Noise and video confirmation episodes differ')
    assets = []
    for key in keys:
        eid = '_'.join(key.split('_')[:-2])
        latent,video,annotation = Path(GT_LATENT_DIR)/(key+'.pt'),Path(GT_VIDEO_DIR)/(key+'.mp4'),Path(ANNOT_DIR)/(eid+'.json')
        if digest(latent)!=confirmed[key]['latent_sha256'] or digest(annotation)!=confirmed[key]['annotation_sha256']:
            raise ValueError(f'Confirmation data changed: {key}')
        assets.extend((latent,video,annotation))
    base = uniform(proposal['per_module_fractions'],protocol['budget'])
    spec = dict(protocol=protocol,provenance=provenance,policy=proposal,
        noise_confirmation=digest(root/'noise_validation.json'),inputs=file_manifest(assets),
        decoder_scheduler=file_manifest([args.vae_model_path,args.scheduler_path]))
    run,cached = prepare_video_run(root/'video_validation',spec)
    output = run/'evaluation'
    for arm,candidate in (('uniform',base),('proposal',proposal)):
        if cached:
            continue
        path = run/('video_'+arm+'_config.json')
        save(path,candidate)
        env = dict(runtime_environment(recipe),PYTHONPATH='.',EVAL_PAIR_SEED=str(protocol['video_seed']),
            SC_SEARCH_RECIPE=json.dumps(recipe.to_dict()),
            SC_MP_GROUP_CHUNK_D=str(recipe.chunk),
            SC_SMOOTH_SCALES=str(Path(protocol['scales']).resolve()),
            SC_MP_TRACE=str(run/('video_'+arm+'_cost.json')),SC_MP_TRACE_MODE='summary',
            SC_MP_PER_MODULE=str(path),SC_MP_CONFIG=json.dumps({k:candidate[k] for k in ('stoc_len_levels','level_fractions')}),
            EVAL_OUT_ROOT=str(output))
        subprocess.run([sys.executable,'evaluate/eval_local_n_samples.py','--config',protocol['config'],
            '--skip',protocol['skip'],'--tag',arm,'--keys_file',str(keys_file),'--num_samples',str(len(keys)),
            '--inference_steps',str(recipe.video_steps),'--scheduler',recipe.video_scheduler],env=env,check=True)
    means,costs = {},{}
    for arm in ('uniform','proposal'):
        trace = json.load(open(run/('video_'+arm+'_cost.json')))
        groups = [g for g in trace['groups'] if g['granularity']=='per_row' and g['chunk_d']==recipe.chunk]
        if not groups or not sum(g['macs'] for g in groups):
            raise ValueError('Missing actual video linear budget trace')
        costs[arm] = sum(g['macs']*g['stoc_len'] for g in groups)/sum(g['macs'] for g in groups)
        rows = [json.load(open(output/arm/'metrics'/(key+'.json'))) for key in keys]
        if any(d['key']!=key or d['generation_seed']!=protocol['video_seed'] for key,d in zip(keys,rows)):
            raise ValueError('Video metric identity or generation seed differs')
        if any(not math.isfinite(d[m]) for d in rows for m in ('psnr','ssim','latent_l2')):
            raise ValueError('Nonfinite video metrics')
        means[arm] = {m:sum(d[m] for d in rows)/len(rows) for m in ('psnr','ssim','latent_l2')}
    status = confirmation_budget_status(costs,protocol['budget'],protocol['budget_tol'],noise['budget_ok'])
    report = dict(keys=keys,means=means,avg_cycles=costs,
        budget_ok=all(status['video_budget_ok'].values()),confirmation_status=status,
        target_cycles=protocol['budget'],budget_tol=protocol['budget_tol'],video_seed=protocol['video_seed'],
        budget_error={arm:cost-protocol['budget'] for arm,cost in costs.items()},
        rule='frozen calibration policy; paired video metrics and measured budget',n10_started=False)
    if not cached:
        save(run/'confirmation.json',report)
        complete_video_run(run)
    save(root/'video_confirmation.json',dict(report,run_directory=str(run),reused=cached))
    finish_confirmation(root,report)
    print(f"Video confirmation complete: status={status['status']}, reused={cached}.",flush=True)


if __name__=='__main__':
    main()
