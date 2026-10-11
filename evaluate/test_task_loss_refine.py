import copy
import unittest
import numpy as np
from evaluate.task_loss_refine import canonical,key,profile_weights,refine,select_on_search,threshold_proposals,threshold_uniform,within_budget
from evaluate.upstream_joint_core import project_levels_to_budget,propose_floor_exchange,_shift_one_boundary

class TaskLossRefineTest(unittest.TestCase):
    def setUp(self):
        self.buckets = {'a':'qkv:l0','b':'mlp_fc1:l1'}
        self.base = dict(stoc_len_levels=[64,96,128],level_fractions=[1,0,0],
            per_module_fractions={n:dict(group_thresholds=[.3,.7],invert=False) for n in self.buckets})
        self.score = dict(mean=1,avg_cycles=96,level_weights=[.3,.4,.3],
            metric_profile={b:[1]*257 for b in self.buckets.values()})

    def test_integer_ladder_preserves_topology(self):
        c = copy.deepcopy(self.base)
        c['stoc_len_levels'] = [65,95,127]
        self.assertEqual(canonical(c)['stoc_len_levels'],[65,95,127])
        for bad in ([0,96,128],[64,96,129],[96,64,128],[64,64,128]):
            c['stoc_len_levels'] = bad
            with self.assertRaises(ValueError): canonical(c)

    def test_parent_schema_rejects_unsupported_module_fields(self):
        for field,value in (('protected',[0]),('protected',[]),('protect_sl',128),
                            ('thresholds',[.1,.2]),('level_fractions',[.2,.4,.4]),('unknown',True)):
            c = copy.deepcopy(self.base)
            c['per_module_fractions']['a'][field] = value
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                canonical(c)

    def test_parent_schema_rejects_wrong_dispatch_and_missing_data(self):
        for c in (dict(self.base,dispatch='per_row'),dict(self.base,protected=[0]),
                  dict(self.base,per_module_fractions={}),dict(self.base,stoc_len_levels=None)):
            with self.assertRaises(ValueError): canonical(c)
        c = copy.deepcopy(self.base)
        del c['per_module_fractions']['a']['group_thresholds']
        with self.assertRaises(ValueError): canonical(c)

    def test_schema_validation_preserves_input(self):
        original = copy.deepcopy(self.base)
        self.assertEqual(canonical(self.base),original)
        self.assertEqual(self.base,original)

    def test_projection_integer_closest(self):
        p = project_levels_to_budget([128,96,64],[.3,.4,.3],target_cost=98)
        self.assertAlmostEqual(p['predicted_cost'],98,delta=.31)
        self.assertTrue(any(L not in [8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128] for L in p['levels']))
        p = project_levels_to_budget([128,96,64],[.3,.4,.3],target_cost=94)
        self.assertAlmostEqual(p['predicted_cost'],94,delta=.31)

    def test_floor_exchange_step_and_donor(self):
        p = propose_floor_exchange([128,96,64],[.3,.4,.3],target_cost=96)
        self.assertEqual(p['receiver_levels'],[128,96,68])
        self.assertEqual(p['levels'],[124,96,68])
        self.assertEqual(len(p['donor_moves']),4)

    def test_threshold_only_active_upstream_specs(self):
        proposals = threshold_proposals(self.base,self.score['metric_profile'],self.buckets)
        self.assertEqual([p[0] for p in proposals],['promote_mlp_floor','promote_attn_linear_floor'])
        for _,c,_ in proposals:
            self.assertLess(min(d['group_thresholds'][0] for d in c['per_module_fractions'].values()),.3)
            self.assertEqual(c['stoc_len_levels'],self.base['stoc_len_levels'])

    def test_exact_runtime_not_profile_accepts(self):
        count = []
        def score(c):
            count.append(c)
            return dict(self.score,mean=.1,avg_cycles=96.4)
        final,_,history = refine(self.base,self.score,score,self.buckets)
        self.assertEqual(final,canonical(self.base))
        self.assertTrue(count)
        self.assertFalse(any(h['accepted'] for h in history))

    def test_repair_can_worsen_loss_and_still_continue(self):
        start = dict(self.score,avg_cycles=90,level_weights=[.5,.2,.3])
        def score(c):
            return dict(self.score,mean=2,avg_cycles=96)
        final,s,history = refine(self.base,start,score,self.buckets,ladder_rounds=0,threshold_rounds=0)
        self.assertEqual(history[0]['stage'],'budget_projection')
        self.assertTrue(history[0]['accepted'])
        self.assertEqual(s['mean'],2)
        self.assertNotEqual(key(final),key(self.base))

    def test_stage_order_and_no_proxy_screen(self):
        def score(c):
            return dict(self.score,mean=sum(c['stoc_len_levels'])/400,avg_cycles=96)
        _,_,history = refine(self.base,self.score,score,self.buckets)
        stages = [h['stage'] for h in history]
        self.assertEqual(stages[:2],['ladder','ladder'])
        self.assertTrue(all(s=='threshold' for s in stages[2:]))

    def test_uniform_reference_retains_topology(self):
        c = threshold_uniform(self.buckets)
        final,_,history = refine(c,self.score,lambda _:self.fail(),self.buckets)
        self.assertEqual(final,c)
        self.assertEqual(history,[])

    def test_shared_bucket_and_profile_weights(self):
        buckets = dict(self.buckets,a2='qkv:l0')
        c = copy.deepcopy(self.base)
        c['per_module_fractions']['a2'] = copy.deepcopy(c['per_module_fractions']['a'])
        for _,p,_ in threshold_proposals(c,self.score['metric_profile'],buckets):
            self.assertEqual(p['per_module_fractions']['a'],p['per_module_fractions']['a2'])
        weights = profile_weights(self.base,self.score['metric_profile'],self.buckets)
        self.assertAlmostEqual(sum(weights),1)

    def test_budget_band(self):
        for value in (95.65,96,96.2,96.35): self.assertTrue(within_budget(value))
        for value in (95.649,96.351,float('nan'),float('inf')): self.assertFalse(within_budget(value))

    def test_selection_frozen_on_search(self):
        rows = [dict(index=0,feasible=True,calibration=dict(mean=1),candidate=threshold_uniform(self.buckets)),
                dict(index=1,feasible=True,calibration=dict(mean=.9),candidate=self.base)]
        self.assertEqual(select_on_search(rows)['index'],1)
        rows[0]['holdout'] = dict(mean=.01)
        rows[1]['holdout'] = dict(mean=10)
        self.assertEqual(select_on_search(rows)['index'],1)

    def test_tied_threshold_matches_prc_dispatch(self):
        shifted,action = _shift_one_boundary([.5],[1,30,1],[0,.5,1],
            boundary=-1,mass_delta=.02,inclusive_high=False)
        self.assertEqual(shifted,[.5])
        self.assertFalse(action['changed'])

    def test_reject_inconsistent_shared_bucket(self):
        buckets = dict(self.buckets,a2='qkv:l0')
        c = copy.deepcopy(self.base)
        c['per_module_fractions']['a2'] = dict(group_thresholds=[.4,.7])
        with self.assertRaises(ValueError):
            profile_weights(c,self.score['metric_profile'],buckets)

if __name__=='__main__':
    unittest.main()
