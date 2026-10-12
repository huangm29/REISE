import tempfile
import unittest
from pathlib import Path
from tr_xas.common import new_output, read_paths, write_json
from tr_xas.cli import validate_config, build_parser
from tr_xas.train_forward import resolve_config


class CoreTests(unittest.TestCase):
    def test_train_dry_by_default(self):
        args = build_parser().parse_args(["train-tcn", "--config", "a", "--paths", "b", "--out", "c"])
        self.assertFalse(args.execute)

    def test_private_paths_relative_to_file(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "paths.json"
            write_json(p, {"mep": "data", "demo": "CHANGE_ME", "fg26": None})
            values = read_paths(p)
            self.assertEqual(values["mep"], Path(root).resolve() / "data")
            self.assertIsNone(values["demo"])
            self.assertIsNone(values["fg26"])

    def test_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "result"
            new_output(out)
            with self.assertRaises(FileExistsError):
                new_output(out)
            with self.assertRaises(ValueError):
                new_output(Path(root) / "nested", [root])

    def test_invalid_epochs(self):
        with self.assertRaises(ValueError):
            validate_config({"seeds": [0], "batch_size": 8, "learning_rate": .001, "epochs": 0}, "tcn")

    def test_forward_isolated_cache_and_paths(self):
        from tr_xas.common import ROOT, read_json
        cfg = read_json(ROOT / "configs/forward_finetune.json")
        paths = {"tr1x": Path("/dataset"), "fg26_checkpoint": Path("/weight.pt")}
        result = resolve_config(cfg, paths, Path("/new_run"))
        self.assertEqual(result["dataset"]["root"], str(Path("/new_run/graph_cache")))
        self.assertEqual(result["optimizer"]["lr"], 1e-5)
        self.assertEqual(result["lambda_mol"], .5)
        self.assertTrue(result["finetune"])


if __name__ == "__main__":
    unittest.main()

