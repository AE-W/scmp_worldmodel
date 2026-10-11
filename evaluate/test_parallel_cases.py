"""CPU checks for ordered parallel aggregation and global curve weights."""
import unittest
import numpy as np
from unittest.mock import patch
from evaluate.parallel_cases import merge_scores,CaseWorkers


class ParallelCasesTest(unittest.TestCase):
    def test_weighted_scores_and_profiles(self):
        results = [dict(per_case=[1],per_case_cycles=[90],level_weights=[1,0],metric_profile={'a':[2,0]}),
                   dict(per_case=[3],per_case_cycles=[100],level_weights=[0,1],metric_profile={'a':[0,4]})]
        result = merge_scores(results,[1,3],profile=True)
        self.assertEqual(result['per_case'],[1,3])
        self.assertEqual(result['mean'],2.5)
        self.assertEqual(result['avg_cycles'],97.5)
        np.testing.assert_allclose(result['level_weights'],[.25,.75])
        np.testing.assert_allclose(result['metric_profile']['a'],[.5,3])

    def test_measurement_tasks_use_global_normalizer(self):
        worker = CaseWorkers.__new__(CaseWorkers)
        tasks = []
        class Pool:
            def map(self,fn,items):
                tasks.extend(items)
                return [({'bucket':[{'case':i}]},{'module':'bucket'}) for i in range(len(items))]
        worker.pool = Pool()
        cases = [dict(key='a',draw=0,weight=1),dict(key='b',draw=1,weight=3)]
        records,buckets = worker.measure(cases,{},64)
        self.assertEqual([t[3]['total_weight'] for t in tasks],[4,4])
        self.assertEqual([r['case'] for r in records['bucket']],[0,1])
        self.assertEqual(buckets,{'module':'bucket'})

    def test_invalid_aggregation_rejected(self):
        with self.assertRaises(ValueError): merge_scores([],[])


if __name__=='__main__':
    unittest.main()
