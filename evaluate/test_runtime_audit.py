"""CPU regressions for hook lifetime and hardware provenance."""
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import patch
from evaluate.runtime_audit import RuntimePatch
from evaluate.search_io import hardware_versions


class RuntimeAuditTest(unittest.TestCase):
    def test_patch_restored_on_normal_exit_and_exception(self):
        target = SimpleNamespace(forward=lambda:1)
        original = target.forward
        for fail in (False,True):
            try:
                with RuntimePatch(target,'forward',lambda:2):
                    self.assertEqual(target.forward(),2)
                    if fail: raise ValueError('simulated failure')
            except ValueError:
                self.assertTrue(fail)
            self.assertIs(target.forward,original)

    def test_exclusive_ownership_and_sequential_use(self):
        target = SimpleNamespace(forward=lambda:1)
        original = target.forward
        first = RuntimePatch(target,'forward',lambda:2)
        first.__enter__()
        try:
            with self.assertRaises(RuntimeError):
                RuntimePatch(target,'forward',lambda:3).__enter__()
            self.assertEqual(target.forward(),2)
        finally:
            first.close()
        first.close()
        self.assertIs(target.forward,original)
        with RuntimePatch(target,'forward',lambda:3):
            self.assertEqual(target.forward(),3)
        self.assertIs(target.forward,original)

    def mock_torch(self, available=True):
        props = SimpleNamespace(name='Test GPU',major=8,minor=0,total_memory=1000)
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:available,
            get_device_properties=lambda device:props),version=SimpleNamespace(cuda='12.4'),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda:90100)))

    def test_hardware_identity_includes_runtime_and_driver(self):
        with patch('evaluate.search_io.subprocess.check_output',return_value='550.54\n550.54\n'):
            result = hardware_versions(self.mock_torch())
        self.assertEqual(result,dict(gpu_name='Test GPU',compute_capability=[8,0],
            total_memory=1000,cuda_runtime='12.4',cudnn_version=90100,driver_version='550.54'))

    def test_driver_unavailable_is_explicit(self):
        with patch('evaluate.search_io.subprocess.check_output',side_effect=subprocess.TimeoutExpired('nvidia-smi',10)), \
             patch('evaluate.search_io.Path.is_file',return_value=False):
            self.assertIsNone(hardware_versions(self.mock_torch())['driver_version'])

    def test_cuda_is_required_for_hardware_identity(self):
        with self.assertRaises(RuntimeError): hardware_versions(self.mock_torch(False))


if __name__=='__main__':
    unittest.main()
