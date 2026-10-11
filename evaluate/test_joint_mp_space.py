import unittest
from evaluate.joint_mp_space import budget_cost, uniform


class JointSpaceTest(unittest.TestCase):
    def setUp(self):
        self.shapes = {'a':[(8,128,16)],'b':[(8,256,16)]}

    def test_uniform(self):
        self.assertEqual(budget_cost(uniform(self.shapes),self.shapes),96)

    def test_cross_module_budget(self):
        c = dict(stoc_len_levels=[128,80],per_module_fractions={
            'a':dict(level_fractions=[1,0]),'b':dict(level_fractions=[0,1])})
        self.assertEqual(budget_cost(c,self.shapes),96)

    def test_thresholds_need_real_inputs(self):
        c = dict(per_module_fractions={'a':dict(group_thresholds=[.5])})
        with self.assertRaises(ValueError):
            budget_cost(c,self.shapes)

    def test_chunk_is_configurable(self):
        shapes = {'a':[(1,192,16)]}
        self.assertEqual(budget_cost(uniform(shapes),shapes,chunk=64),96)
        with self.assertRaises(ValueError): budget_cost(uniform(shapes),shapes)
        for chunk in (0,-1,64.5,True):
            with self.assertRaises(ValueError): budget_cost(uniform(shapes),shapes,chunk)

    def test_fractional_budget_uses_requested_group_count(self):
        shapes = {'a':[(1,192,16)]}
        candidate = dict(stoc_len_levels=[128,64],per_module_fractions={
            'a':dict(level_fractions=[1/3,2/3])})
        self.assertAlmostEqual(budget_cost(candidate,shapes,chunk=64),256/3)


if __name__ == '__main__':
    unittest.main()
