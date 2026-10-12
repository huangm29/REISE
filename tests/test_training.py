import unittest
import numpy as np
import torch
from tr_xas.common import encoder_modules, temporal_module


class TrainingSmokeTests(unittest.TestCase):
    def test_original_encoder_training_step(self):
        try:
            import scipy
            import sklearn
        except ImportError:
            self.skipTest("Training dependencies scipy/sklearn are not installed in this environment")
        _, models, train = encoder_modules()
        torch.set_num_threads(2)
        rng = np.random.default_rng(123)
        arrays = {
            "rxn_ids": ["synthetic_a", "synthetic_b"],
            "XR": rng.normal(size=(2, 3, 200)).astype(np.float32),
            "XP": rng.normal(size=(2, 3, 200)).astype(np.float32),
            "XK": rng.normal(size=(2, 10, 3, 200)).astype(np.float32),
            "PR": np.ones((2, 3), np.float32), "PP": np.ones((2, 3), np.float32),
            "PK": np.ones((2, 10, 3), np.float32), "comp": np.ones((2, 7), np.float32),
            "labels": {key: np.zeros((2, 10), np.float32) for key in train.LABEL_KEYS},
        }
        arrays["labels"]["s"][:] = np.linspace(0, 1, 10)
        model = models.B3Model()
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        loss = train.run_epoch(model, "B3", arrays, np.array([0, 1]), "cpu", opt)
        self.assertTrue(np.isfinite(loss))
        val = train.run_epoch(model, "B3", arrays, np.array([2, 3]), "cpu")
        self.assertTrue(np.isfinite(val))

    def test_original_temporal_training_step(self):
        before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic", before[0])
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark", before[1])
        module = temporal_module("unused_synthetic_test")
        module.EPOCHS, module.BATCH_SIZE = 1, 2
        rng = np.random.default_rng(123)
        x = rng.normal(size=(2, 201, 12)).astype(np.float32)
        ds = np.tile(np.abs(np.linspace(-1, 1, 201)), (2, 1)).astype(np.float32)
        y = (ds < .1).astype(np.float32)
        model, history = module.train_seed(0, x, y, ds, np.zeros(12, np.float32), np.ones(12, np.float32), torch.device("cpu"), "synthetic_smoke")
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(history[0]["total"]))
        raw, spread = module.score_models([model], x, np.zeros(12, np.float32), np.ones(12, np.float32), torch.device("cpu"))
        self.assertEqual(raw.shape, (2, 201))
        self.assertTrue(np.isfinite(raw).all())


if __name__ == "__main__":
    unittest.main()
