import unittest
import numpy as np
import torch
from tr_xas.common import encoder_modules, temporal_module, scoring_module
from tr_xas.inference import validate_input


class ModelTests(unittest.TestCase):
    def test_encoder_and_missing_edges(self):
        torch.set_num_threads(2)
        _, models = encoder_modules(training=False)
        model = models.B3Model().eval()
        x = torch.zeros(2, 3, 3, 200)
        p = torch.tensor([[[1., 0., 1.]] * 3] * 2)
        out = model(x, p, torch.ones(2, 7))
        self.assertEqual(out["s_logits"].shape, (2, 20))
        self.assertEqual(out["prepost_logits"].shape, (2, 3))
        self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
        sum(v.square().mean() for v in out.values()).backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_tcn_train_score_architecture_parity(self):
        train = temporal_module("unused_test_path")
        score = scoring_module()
        a, b = train.TemporalDetector().eval(), score.TemporalDetector().eval()
        b.load_state_dict(a.state_dict(), strict=True)
        x = torch.randn(2, 12, 201)
        torch.testing.assert_close(a(x), b(x), atol=0, rtol=0)
        probabilities = torch.sigmoid(a(x)).detach().numpy()
        for i in range(2):
            np.testing.assert_array_equal(train.persistent9(probabilities)[i], score.persistent9(probabilities[i]))

    def test_temporal_losses_and_backward(self):
        train = temporal_module("unused_test_path")
        logits = torch.randn(2, 201, requires_grad=True)
        ds = torch.linspace(0, 1, 201).repeat(2, 1)
        labels = (ds < .1).float()
        loss = train.rank_loss(logits, labels) + .25 * train.listwise_center_loss(logits, ds) + train.expected_distance_loss(logits, ds)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_wrong_sampling_rejected(self):
        data = {"query": np.zeros((201, 3, 200)), "endpoint": np.zeros((2, 3, 200)), "composition": np.array([.5, .5, 0, 0, 1, 0, 0]), "time_fs": np.arange(201) * 2}
        with self.assertRaises(ValueError):
            validate_input(data)
        data["time_fs"] = np.arange(201)
        validate_input(data)


if __name__ == "__main__":
    unittest.main()
