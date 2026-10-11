"""CPU checks for confirmation eligibility and superseded artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
from evaluate.confirmation_status import begin_confirmation,confirmation_budget_status,finish_confirmation
from evaluate.search_io import digest,save


class ConfirmationStatusTest(unittest.TestCase):
    def test_paired_budgets_and_noise_confirmation(self):
        status = confirmation_budget_status({'uniform':96,'proposal':96.2},96,.35,True)
        self.assertTrue(status['evaluation_complete'])
        self.assertTrue(status['passed'])
        self.assertEqual(status['status'],'passed')
        for costs,noise in (({'uniform':96,'proposal':97},True),
                            ({'uniform':97,'proposal':96},True),
                            ({'uniform':96,'proposal':96},False),
                            ({'uniform':96,'proposal':float('nan')},True)):
            with self.subTest(costs=costs,noise=noise):
                status = confirmation_budget_status(costs,96,.35,noise)
                self.assertTrue(status['evaluation_complete'])
                self.assertFalse(status['passed'])
                self.assertEqual(status['status'],'budget_failed')

    def test_pending_invalidates_previous_pass_and_archives_policy_only_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root/'selected.json',{'old':True})
            save(root/'search_policy.json',{'frozen':True})
            save(root/'confirmation_status.json',{'passed':True})
            begin_confirmation(root)
            self.assertFalse((root/'selected.json').exists())
            archived = list((root/'superseded').glob('selected-*.json'))
            self.assertEqual(len(archived),1)
            self.assertEqual(json.loads(archived[0].read_text()),{'old':True})
            status = json.loads((root/'confirmation_status.json').read_text())
            self.assertFalse(status['passed'])
            self.assertFalse(status['evaluation_complete'])
            self.assertEqual(json.loads((root/'search_policy.json').read_text()),{'frozen':True})

    def test_failed_confirmation_preserves_frozen_policy_and_does_not_select(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root/'search_policy.json',{'frozen':True})
            policy_hash = digest(root/'search_policy.json')
            begin_confirmation(root)
            status = confirmation_budget_status({'uniform':96,'proposal':97},96,.35,True)
            finish_confirmation(root,{'confirmation_status':status})
            saved = json.loads((root/'confirmation_status.json').read_text())
            self.assertFalse(saved['passed'])
            self.assertEqual(saved['policy_sha256'],policy_hash)
            self.assertEqual(digest(root/'search_policy.json'),policy_hash)
            self.assertFalse((root/'selected.json').exists())

    def test_missing_arm_and_non_boolean_noise_status_rejected(self):
        for costs,noise in (({'proposal':96},True),({'uniform':96,'proposal':96},'false')):
            with self.assertRaises(ValueError): confirmation_budget_status(costs,96,.35,noise)


if __name__=='__main__':
    unittest.main()
