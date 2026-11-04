"""Selected comparison, canonical v2 naming and the exact CSRv2 curriculum."""
import ast
import contextlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest import mock

import mmrsae_imagenet as runner
import mmrsae_reporting as reporting
from mmrsae import MMRSAEConfig, cosine_annealed_k
from mmrsae_methods import BENCHMARK_METHODS


_MSYS_BASH = Path("C:/msys64/usr/bin/bash.exe")
_BASH = str(_MSYS_BASH) if os.name == "nt" and _MSYS_BASH.exists() else shutil.which("bash")


class SelectedBenchmarkTests(unittest.TestCase):
    def test_cli_and_report_select_only_the_four_final_methods(self):
        expected = ("csr", "csrv2", "mpsae", "mpsaev2")
        self.assertEqual(BENCHMARK_METHODS, expected)
        self.assertEqual(runner.VARIANTS, expected)
        self.assertEqual(reporting.METHODS, expected)
        parser = runner.build_parser()
        self.assertEqual(parser.parse_args([]).variant, "mpsae")
        for method in ("mmrsae", "mmrsae-rel", "mmrsae-wide", "mmrsae-annealed", "dlmmsae", "dlmmsaev2"):
            with self.subTest(method=method), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--variant", method])

    def test_old_name_is_canonicalized_before_config_and_plan_serialization(self):
        old = MMRSAEConfig(input_dim=6, hidden_dim=128, variant="mpsae-annealed")
        new = MMRSAEConfig(input_dim=6, hidden_dim=128, variant="mpsaev2")
        self.assertEqual(old.to_dict(), new.to_dict())
        self.assertEqual(old.variant, "mpsaev2")
        output = io.StringIO()
        with mock.patch("mmrsae_data.prepare_imagenet_features", side_effect=AssertionError("data opened")), \
                mock.patch("torch.cuda.init", side_effect=AssertionError("CUDA initialized")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(["--plan", "--variant", "mpsae-annealed"]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["variant"], "mpsaev2")
        self.assertEqual(plan["settings"]["variant"], "mpsaev2")
        self.assertFalse(plan["context_distillation"])
        self.assertEqual(plan["sparsity_schedule"]["type"], "fixed-main-linear-ot")

    def test_v2_budgets_match_the_original_csrv2_formula_at_every_step(self):
        # Execute only the actual reference function, without importing the
        # historical runner's dataset/backbone dependencies or starting jobs.
        path = Path(__file__).resolve().parents[1] / "csr_vs_mmpot_imagenet.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "cosine_annealed_k")
        namespace = {"math": math}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        reference = namespace["cosine_annealed_k"]
        for dimension in (512, 2048):
            for total, start, target, fraction in ((500, 64, 8, .7), (333, 63, 7, .37),
                    (10, 32, 8, 1.), (2, 64, 8, .7), (1, 64, 8, .7), (100, 8, 8, .7)):
                settings = dict(input_dim=dimension, k=target, anneal_start_k=start, anneal_fraction=fraction)
                csr = MMRSAEConfig(**settings, variant="csrv2")
                dlm = MMRSAEConfig(**settings, variant="dlmmsaev2")
                for step in range(total):
                    expected = reference(step, total, start, target, fraction)
                    self.assertEqual(csr.training_k(step, total), expected)
                    self.assertEqual(dlm.training_k(step, total), expected)
        with self.assertRaises(ValueError):
            cosine_annealed_k(0, 0, 64, 8)


@unittest.skipUnless(_BASH, "Bash is required for launcher selection checks")
class LauncherSelectionTests(unittest.TestCase):
    def launch(self, *arguments, selection=None):
        root = Path(__file__).resolve().parents[1]
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("MMRSAE_", "WANDB_")) and key != "ENV_FILE"}
        environment.update(PYTHON=Path(sys.executable).as_posix(), WANDB_MODE="offline")
        if os.name == "nt":
            # A Windows Python environment need not expose MSYS coreutils.
            environment["PATH"] = str(Path(_BASH).parent) + os.pathsep + environment.get("PATH", "")
        if selection is not None:
            environment["MMRSAE_VARIANTS"] = selection
        return subprocess.run([_BASH, (root / "run_mmrsae_imagenet.sh").as_posix(),
                               "--plan", *arguments], cwd=root, env=environment,
                              capture_output=True, text=True, timeout=60)

    def variants(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        remaining, plans = result.stdout.strip(), []
        decoder = json.JSONDecoder()
        while remaining:
            plan, offset = decoder.raw_decode(remaining)
            self.assertEqual(plan["action"], "plan only; no downloads, GPU initialization or training")
            plans.append(plan["variant"])
            remaining = remaining[offset:].strip()
        return plans

    def test_launcher_without_method_settings_trains_only_the_new_methods(self):
        # csr/csrv2 come from the published baseline fits, fetched before the report.
        self.assertEqual(self.variants(self.launch()), ["mpsae", "mpsaev2"])

    def test_explicit_list_can_still_train_every_benchmark_method(self):
        self.assertEqual(self.variants(self.launch(selection=",".join(BENCHMARK_METHODS))), list(BENCHMARK_METHODS))

    def test_explicit_cli_selection_overrides_list_once_in_both_forms(self):
        selection = ",".join(BENCHMARK_METHODS)
        self.assertEqual(self.variants(self.launch("--variant", "mpsaev2", selection=selection)), ["mpsaev2"])
        self.assertEqual(self.variants(self.launch("--variant=mpsae-annealed", selection=selection)), ["mpsaev2"])

    def test_unsupported_list_is_rejected_before_any_method_runs(self):
        result = self.launch(selection="csr,dlmmsae,mpsae")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("Unsupported benchmark method", result.stderr)


if __name__ == "__main__":
    unittest.main()
