"""CPU structural regressions for the explicit experiment stages."""
import ast
from pathlib import Path
import tempfile
import unittest
from evaluate.search_io import digest,save

ROOT = Path(__file__).resolve().parents[1]


class StageStructureTest(unittest.TestCase):
    def test_threshold_constructor_is_not_shadowed(self):
        tree = ast.parse((ROOT/'evaluate/joint_mp_search.py').read_text())
        main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        writes = [n.id for n in ast.walk(main) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)]
        self.assertNotIn('threshold_uniform',writes)

    def test_active_scripts_have_no_retired_imports(self):
        for name in ('joint_mp_search.py','joint_mp_video_confirm.py'):
            tree = ast.parse((ROOT/'evaluate'/name).read_text())
            imports = [n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            self.assertNotIn('evaluate.calibrate_group_mp',imports)
            self.assertNotIn('evaluate.joint_mp_video_gate',imports)

    def test_frozen_selection_written_with_completion(self):
        source = (ROOT/'evaluate/joint_mp_search.py').read_text()
        self.assertIn("state['selected_index'] = best['index']",source)
        self.assertNotIn('noise_selected.json',source)
        self.assertNotIn('validation_indices',source)

    def test_mixed_runtime_check_is_in_smoke_stage(self):
        tree = ast.parse((ROOT/'evaluate/joint_mp_search.py').read_text())
        smoke = next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                     and ast.unparse(n.test)=="cli.stage == 'smoke'")
        calls = [n.func.id for n in ast.walk(smoke) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertIn('check_mixed_runtime',calls)

    def test_environment_is_pinned_before_kernel_import(self):
        source = (ROOT/'evaluate/joint_mp_search.py').read_text()
        self.assertLess(source.index('os.environ.update(_runtime_env)'),source.index('from scmp_kernels.mp import'))
        child = (ROOT/'evaluate/eval_local_n_samples.py').read_text()
        self.assertLess(child.index('configure_float_precision(torch)'),child.index('from models import'))

    def test_video_confirmation_uses_schema_and_explicit_status(self):
        source = (ROOT/'evaluate/joint_mp_video_confirm.py').read_text()
        self.assertIn('canonical(proposal,',source)
        self.assertIn('begin_confirmation(root)',source)
        self.assertIn('finish_confirmation(root,report)',source)
        self.assertNotIn("save(root/'selected.json'",source)
        self.assertLess(source.index('canonical(proposal,'),source.index('subprocess.run('))

    def test_evaluator_closed_in_finally(self):
        tree = ast.parse((ROOT/'evaluate/joint_mp_search.py').read_text())
        main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        cleanup = [ast.unparse(n) for t in ast.walk(main) if isinstance(t,ast.Try) for n in t.finalbody]
        self.assertIn('evaluator.close()',cleanup)

    def test_hardware_is_recorded_and_checked(self):
        source = (ROOT/'evaluate/joint_mp_search.py').read_text()
        self.assertIn('hardware_versions=hardware_versions(torch,evaluator.device)',source)
        child = (ROOT/'evaluate/joint_mp_video_confirm.py').read_text()
        self.assertIn("hardware_versions(torch)!=provenance['hardware_versions']",child)

    def test_atomic_json_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            save(path,{'complete':False})
            old = digest(path)
            save(path,{'complete':True,'selected_index':0})
            self.assertNotEqual(old,digest(path))
            self.assertFalse(path.with_suffix('.json.tmp').exists())


if __name__=='__main__':
    unittest.main()
