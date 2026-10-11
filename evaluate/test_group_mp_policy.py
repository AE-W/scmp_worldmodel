import unittest
from evaluate.group_mp_policy import mean_cycles


class GroupPolicyTest(unittest.TestCase):
    def test_exact_rounding(self):
        self.assertEqual(mean_cycles(4, [128, 96, 64], [.5, 0, .5]), 96)
        self.assertEqual(mean_cycles(3, [128, 96, 64], [.5, 0, .5]), 320 / 3)
        for n in (1, 3, 17, 1000):
            self.assertEqual(mean_cycles(n, [128, 112, 96, 64, 32], [0, 0, 1, 0, 0]), 96)


if __name__ == "__main__":
    unittest.main()
