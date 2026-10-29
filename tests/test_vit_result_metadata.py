"""Check reporting without importing GPU or dataset dependencies."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_reporting(variant):
    path = ROOT / "ViT" / f"GMU_{variant}_ViT_finetuning_Code.py"
    tree = ast.parse(path.read_text())
    namespace = {"os": os}
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.isupper()
            for target in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        elif isinstance(node, ast.FunctionDef) and node.name == "experiment_label":
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    trial = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_trial")
    return tree, trial, namespace


def history(trial, namespace):
    node = next(n for n in trial.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "history" for t in n.targets))
    return eval(compile(ast.Expression(node.value), "<history>", "eval"), namespace)


class ResultMetadataTests(unittest.TestCase):
    def test_default_metadata_matches_application_records(self):
        for variant, ratio, layers in (
            ("Equiv", 0.0, [0, 1, 6, 7, 8, 9, 10, 11]),
            ("PseudoEquiv", 0.05, list(range(12))),
        ):
            with self.subTest(variant=variant):
                _, trial, ns = load_reporting(variant)
                self.assertEqual(ns["GMU_TARGET_LAYERS"], [2, 3, 4, 5])
                self.assertEqual(ns["NATIVE_CAMO_RATIO"], ratio)
                self.assertEqual(ns["NATIVE_CAMO_TARGET_LAYERS"], layers)
                native = {i: [0] for i in layers} if ratio > 0 else {}
                ns.update(dataset_name="cifar10", trial_idx=0, seed=1,
                          native_camo_indices=native, native_camo_stats={},
                          qk_camo_indices={i: [0] for i in range(12)},
                          qk_camo_stats={}, initial_val_acc=0.5)
                result = history(trial, ns)
                self.assertEqual(result["native_camouflage"]["enabled"], bool(native))
                self.assertEqual(result["native_camouflage"]["explicitly_excludes_gmu_layers"],
                                 not bool(set(layers) & {2, 3, 4, 5}))
                self.assertEqual(result["native_camouflage"]["target_layers"], layers)
                self.assertEqual(result["model_type"], ns["experiment_label"]())
                self.assertTrue(result["qk_camouflage"]["enabled"])

    def test_labels_follow_disabled_disjoint_and_overlapping_settings(self):
        _, _, ns = load_reporting("PseudoEquiv")
        for ratio, layers, suffix in (
            (0, [2], "GMU_with_qk_camouflage"),
            (0.05, [], "GMU_with_qk_camouflage"),
            (0.05, [0, 1], "outside_gmu_layers"),
            (0.05, [1, 2], "overlapping_gmu_layers"),
            (0.05, list(range(12)), "overlapping_gmu_layers"),
        ):
            ns.update(NATIVE_CAMO_RATIO=ratio, NATIVE_CAMO_TARGET_LAYERS=layers)
            self.assertTrue(ns["experiment_label"]().endswith(suffix))

    def test_stats_exclusions_match_supplied_target_layers(self):
        for variant in ("Equiv", "PseudoEquiv"):
            tree, _, ns = load_reporting(variant)
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == "apply_native_scale_camouflage")
            stats = next(n for n in ast.walk(function) if isinstance(n, ast.Dict)
                         and any(isinstance(k, ast.Constant) and k.value == "excluded_gmu_layers"
                                 for k in n.keys))
            expr = next(v for k, v in zip(stats.keys, stats.values) if k.value == "excluded_gmu_layers")
            ns["target_layers"] = ns["NATIVE_CAMO_TARGET_LAYERS"]
            actual = eval(compile(ast.Expression(expr), "<stats>", "eval"), ns)
            self.assertEqual(actual, [2, 3, 4, 5] if variant == "Equiv" else [])

    def test_new_and_legacy_files_both_skip_without_loading_model(self):
        for variant in ("Equiv", "PseudoEquiv"):
            _, trial, ns = load_reporting(variant)
            prefix = trial.body[:next(i for i, n in enumerate(trial.body)
                                     if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                                     and isinstance(n.value.func, ast.Name)
                                     and n.value.func.id == "set_seed")]
            fn = ast.FunctionDef(name="check", args=trial.args, body=prefix + [ast.Return(ast.Name("save_path", ast.Load()))], decorator_list=[])
            module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
            ns["torch"] = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            for existing in ("new", "legacy", None):
                paths, logs = [], []
                def exists(path):
                    paths.append(path)
                    return existing == ("legacy" if "exclude_gmulayers" in path else "new")
                ns.update(os=SimpleNamespace(path=SimpleNamespace(join=os.path.join, exists=exists)),
                          log_print=logs.append)
                exec(compile(module, "<trial>", "exec"), ns)
                result = ns["check"]("cifar10", 1, 0, 0)
                if existing:
                    self.assertIsNone(result)
                    self.assertEqual(len(logs), 1)
                    self.assertIn("[SKIP]", logs[0])
                else:
                    self.assertIn(ns["experiment_label"](), result)
                    self.assertNotIn("bothcamo", result)
                self.assertIn(ns["experiment_label"](), paths[0])

    def test_only_start_skip_and_done_log_calls_remain(self):
        for variant in ("Equiv", "PseudoEquiv"):
            tree, _, _ = load_reporting(variant)
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "log_print"]
            self.assertEqual(len(calls), 3)
            text = " ".join(ast.unparse(n) for n in calls)
            for tag in ("[START]", "[SKIP]", "[DONE]"):
                self.assertIn(tag, text)


if __name__ == "__main__":
    unittest.main()
