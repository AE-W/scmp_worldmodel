import itertools
import unittest
import numpy as np
from evaluate.upstream_mp_solver import DENSE_LADDER, bin_records, policy, solve_global, staircase, thresholds


class UpstreamSolverTest(unittest.TestCase):
    def test_upstream_ladder(self):
        self.assertEqual(DENSE_LADDER,[8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128])

    def test_dp_vs_exhaustive(self):
        rng = np.random.default_rng(3)
        for _ in range(12):
            E,C,L = rng.normal(size=(4,3)),rng.random(4),np.array([8,32,128])
            p = staircase(E,C,L,.03)
            actual = sum(E[b,p[b]]+.03*C[b]*L[p[b]] for b in range(4))
            optimum = min(sum(E[b,r]+.03*C[b]*L[r] for b,r in enumerate(seq))
                          for seq in itertools.combinations_with_replacement(range(3),4))
            self.assertAlmostEqual(actual,optimum)
            self.assertTrue(np.all(np.diff(p)>=0))

    def test_thresholds_deploy_path_including_boundary(self):
        edges = np.array([.1,.4,.7])
        path = np.array([1,1,2,4])
        cuts = thresholds(path,edges,5)
        self.assertEqual(list(np.searchsorted(cuts,[0,.1,.2,.4,.5,.7,.8,1],side='left')),
                         [0,1,1,1,2,2,4,4])  # upstream zero sentinel

    def test_upstream_equal_count_ties_and_sampled_budget(self):
        metric = np.array([0,.5,.5,1])
        err = np.ones((4,17))
        rec = dict(metric=metric,full_metric=np.array([0]*8+[.5]*40+[1]*2),
                   mac_weight=128,population=50,sampled_rows=4,raw=err,fis=err)
        b = bin_records([rec],2048)
        self.assertEqual(b['costs'].sum(),50*128)
        self.assertEqual(b['errors']['raw'].sum(),0)
        self.assertEqual(b['edges'].tolist(),[0,.5,.5,1])

    def test_global_budget_and_realized_thresholds(self):
        records = {}
        for name in ('a','b'):
            m = np.linspace(.001,1,40)
            E = np.outer(m+.1,1/np.asarray(DENSE_LADDER))
            records[name] = bin_records([dict(metric=m,full_metric=m,mac_weight=128,
                                             population=40,sampled_rows=40,raw=E,fis=E)],16)
        cuts,diag = solve_global(records,'fis',96)
        actual = np.mean([np.asarray(DENSE_LADDER)[np.searchsorted(cuts[n],np.linspace(.001,1,40),side='left')].mean()
                          for n in records])
        self.assertAlmostEqual(actual,diag['avg_cycles'])
        self.assertLessEqual(actual,96)
        self.assertTrue(all(np.all(np.diff(v)>=0) for v in cuts.values()))

    def test_uniform_inside_threshold_space(self):
        cuts = [-1 if L<96 else 1 for L in DENSE_LADDER[:-1]]
        indices = np.searchsorted(cuts,np.linspace(0,1,100),side='left')
        self.assertTrue(np.all(np.asarray(DENSE_LADDER)[indices]==96))
        c = policy({'a':'u'},{'u':cuts})
        self.assertEqual(len(c['stoc_len_levels']),17)

    def test_bin_sums_match_upstream_reference(self):
        rng = np.random.default_rng(17)
        parts = []
        for size,weight in ((20,128),(30,256)):
            parts.append(dict(metric=rng.choice([0,.1,.3,.8,1],size),
                population=size*3,sampled_rows=size,mac_weight=weight,
                raw=rng.random((size,17)),fis=rng.random((size,17))))
        actual = bin_records(parts,9)
        metric = np.concatenate([r['metric'] for r in parts])
        order = np.argsort(metric,kind='stable')
        edges = np.linspace(0,len(metric),10).astype(np.int64)
        np.testing.assert_array_equal(actual['edges'],metric[order][edges[1:]-1])
        mac = np.concatenate([np.full(len(r['metric']),r['mac_weight']*3) for r in parts])
        np.testing.assert_array_equal(actual['costs'],np.add.reduceat(mac[order],edges[:-1]))
        for currency in ('raw','fis'):
            error = np.concatenate([r[currency]-r[currency][:,[-1]] for r in parts])
            np.testing.assert_array_equal(actual['errors'][currency],np.add.reduceat(error[order],edges[:-1],axis=0))


if __name__ == '__main__':
    unittest.main()
