"""Regression tests for configured search spaces and confirmation reuse."""
from pathlib import Path
import tempfile
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
import numpy as np
from evaluate.search_recipe import (DEFAULT_RECIPE,resolve_ladder,default_ladder,
    uniform_variants,runtime_environment,runtime_settings,configure_float_precision)
from evaluate.mixed_runtime_smoke import mixed_smoke_layout
from evaluate.upstream_mp_solver import DENSE_LADDER,bin_records,solve_global,policy
from evaluate.task_loss_refine import canonical,refine
from evaluate.search_io import save,file_manifest,prepare_video_run,complete_video_run,runtime_sources


class SearchConfigurationTest(unittest.TestCase):
    def test_non_grid_budget_and_defaults(self):
        self.assertEqual(resolve_ladder(DENSE_LADDER,1,128,95),DENSE_LADDER)
        self.assertEqual(resolve_ladder(DENSE_LADDER,1,128,96),DENSE_LADDER)
        self.assertEqual(default_ladder(DENSE_LADDER,1,128,96),DENSE_LADDER)
        self.assertEqual(default_ladder(DENSE_LADDER,1,128,1)[0],1)
        self.assertEqual(default_ladder(DENSE_LADDER,64,120,120)[-1],120)
        for bounds in ((1,128,129),(64,128,32),(0,128,96),(1,128,95.5)):
            with self.assertRaises(ValueError): resolve_ladder(DENSE_LADDER,*bounds)

    def test_uniform_smoke_for_all_integer_budgets(self):
        for budget in range(1,129):
            ladder = default_ladder(DENSE_LADDER,1,128,budget)
            padded,threshold = uniform_variants(['a'],budget,ladder)
            self.assertEqual(sum(a*b for a,b in zip(padded['stoc_len_levels'],padded['level_fractions'])),budget)
            idx = np.searchsorted(threshold['per_module_fractions']['a']['group_thresholds'],[0,.1,.5,1],side='left')
            self.assertTrue(np.all(np.asarray(threshold['stoc_len_levels'])[idx]==budget))

    def test_custom_ladder_curves_and_solver(self):
        lengths = [40,70,120]
        m = np.linspace(.01,1,100)
        err = np.outer(m,1/np.asarray(lengths))
        record = dict(metric=m,raw=err,fis=err,population=100,sampled_rows=100,mac_weight=128)
        bins = {'a':bin_records([record],20)}
        cuts,diag = solve_global(bins,'fis',95,levels=lengths)
        p = policy({'module':'a'},cuts,lengths)
        self.assertEqual(p['stoc_len_levels'],lengths)
        self.assertLessEqual(diag['avg_cycles'],95)

    def test_explicit_ladder_is_exact(self):
        supplied = [40,70,120]
        self.assertEqual(resolve_ladder(supplied,1,128,95),supplied)
        self.assertEqual(supplied,[40,70,120])
        for values,bounds in (([64,96],(1,128,112)),([64,96],(1,128,32)),
                              ([32,64,128],(64,120,96))):
            with self.assertRaises(ValueError): resolve_ladder(values,*bounds)

    def test_runtime_environment_pins_numerics(self):
        incoming = dict(CUDA_VISIBLE_DEVICES='2',BRIDGE_ROOT='/data',SC_HW_MAX_MASKS='8',
            SC_FORCE_TABLE='1',SC_FORCE_COMPACT='1',SC_DISABLE_OWEN='1',SC_MP_CONFIG='old',
            SC_MP_TRACE='/old/trace.json',NVIDIA_TF32_OVERRIDE='1')
        env = runtime_environment(DEFAULT_RECIPE,incoming)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'2')
        self.assertEqual(env['BRIDGE_ROOT'],'/data')
        self.assertEqual(env['SC_HW_MAX_MASKS'],'64')
        self.assertEqual(env['NVIDIA_TF32_OVERRIDE'],'0')
        for key in ('SC_FORCE_TABLE','SC_FORCE_COMPACT','SC_DISABLE_OWEN','SC_MP_CONFIG','SC_MP_TRACE'):
            self.assertNotIn(key,env)
        self.assertEqual(incoming['SC_HW_MAX_MASKS'],'8')
        for key,value in runtime_settings(DEFAULT_RECIPE).items():
            self.assertEqual(env[key],value)

    def test_float_precision_is_explicit(self):
        modes = []
        mock = SimpleNamespace(set_float32_matmul_precision=modes.append,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True)))
        configure_float_precision(mock)
        self.assertEqual(modes,['highest'])
        self.assertFalse(mock.backends.cuda.matmul.allow_tf32)
        self.assertFalse(mock.backends.cudnn.allow_tf32)

    def test_mixed_smoke_exercises_boundaries_and_all_rungs(self):
        spec = mixed_smoke_layout()
        rungs = np.searchsorted(spec['cuts'],np.asarray(spec['amplitudes']),side='left')
        np.testing.assert_array_equal(rungs,spec['rungs'])
        self.assertEqual(set(rungs.reshape(-1)),{0,1,2})
        self.assertIn(.25,np.asarray(spec['amplitudes']))
        self.assertIn(.75,np.asarray(spec['amplitudes']))
        self.assertIn(79,spec['levels'])

    def test_initial_bounds_are_enforced(self):
        c = dict(stoc_len_levels=[64,96,128],per_module_fractions={'a':dict(group_thresholds=[.3,.7])})
        with self.assertRaises(ValueError): canonical(c,max_level=120)
        with self.assertRaises(ValueError):
            refine(c,{},lambda _:self.fail(),{'a':'qkv:l0'},max_level=120)

    def test_ladder_candidate_bounds_before_evaluation(self):
        c = dict(stoc_len_levels=[117,119,120],per_module_fractions={'a':dict(group_thresholds=[.3,.7])})
        s = dict(mean=1,avg_cycles=119,level_weights=[.2,.4,.4],metric_profile={'qkv:l0':[1]*257})
        final,_,history = refine(c,s,lambda _:self.fail('Out-of-bounds candidate evaluated'),
            {'a':'qkv:l0'},budget=119,max_level=120,ladder_rounds=1,threshold_rounds=0)
        self.assertEqual(final['stoc_len_levels'],[117,119,120])
        self.assertFalse(history[0]['accepted'])

    def test_video_cache_complete_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = dict(seed=3407,checkpoint='hash-a',inputs={'sample':'hash-x'})
            first,cached = prepare_video_run(root,spec)
            self.assertFalse(cached)
            partial,cached = prepare_video_run(root,spec)
            self.assertFalse(cached)
            self.assertNotEqual(first,partial)
            save(partial/'metrics.json',dict(psnr=24))
            complete_video_run(partial)
            reused,cached = prepare_video_run(root,spec)
            self.assertTrue(cached)
            self.assertEqual(reused,partial)
            save(partial/'metrics.json',dict(psnr=100))
            fresh,cached = prepare_video_run(root,spec)
            self.assertFalse(cached)
            self.assertNotEqual(fresh,partial)
            changed = dict(spec,checkpoint='hash-b')
            fresh2,cached = prepare_video_run(root,changed)
            self.assertFalse(cached)
            self.assertNotEqual(fresh,fresh2)

    def test_dataset_manifest_detects_content_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.json'
            save(path,{'x':1})
            old = file_manifest([path])
            save(path,{'x':2})
            self.assertNotEqual(old,file_manifest([path]))

    def test_video_cache_tracks_code_and_numerical_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = dict(runtime_sources={'sample/pipeline.py':'v1'},
                        runtime_settings=runtime_settings(DEFAULT_RECIPE))
            first,_ = prepare_video_run(root,spec)
            save(first/'metrics.json',dict(psnr=24))
            complete_video_run(first)
            for changed in (dict(spec,runtime_sources={'sample/pipeline.py':'v2'}),
                            dict(spec,runtime_settings=dict(spec['runtime_settings'],SC_HW_MAX_MASKS='8')),
                            dict(spec,hardware_versions={'gpu_name':'different GPU','cuda_runtime':'12.4'})):
                fresh,cached = prepare_video_run(root,changed)
                self.assertFalse(cached)
                self.assertNotEqual(fresh,first)

    def test_runtime_sources_include_video_and_kernel(self):
        sources = runtime_sources()
        self.assertIn('evaluate/joint_mp_video_confirm.py',sources)
        self.assertIn('util.py',sources)
        self.assertIn('sample/pipeline_trajectory2videogen.py',sources)
        self.assertIn('dataset/dataset_util.py',sources)
        self.assertIn('evaluate/mixed_runtime_smoke.py',sources)
        self.assertTrue(any(p.startswith('kernels/scmp_kernels/') for p in sources))

    def test_runner_forwards_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory)/'python'
            python.write_text('#!'+sys.executable+'\nimport sys,json\nprint("mlp_fc1=1" if sys.argv[1]=="-c" else json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            env = dict(os.environ,PATH=directory+os.pathsep+os.environ['PATH'])
            result = subprocess.run(['bash','RUN_JOINT_SEARCH.sh','search','--budget','95','--max_level','120'],
                env=env,check=True,capture_output=True,text=True)
            import json
            argv = json.loads(result.stdout)
            self.assertEqual(argv[-4:],['--budget','95','--max_level','120'])
            self.assertEqual(argv[argv.index('--stage')+1],'search')


if __name__=='__main__':
    unittest.main()
