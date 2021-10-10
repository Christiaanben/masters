from unittest import TestCase

import torch

from genbot.model import ScoreThreshold


class ScoreThresholdTestCase(TestCase):

    def test_score_threshold_untrained(self):
        threshold = ScoreThreshold()
        self.assertEqual(threshold.midpoint, 0.5)
        classified_outputs = threshold.classify_outputs(torch.tensor([[0.1, 0.2, .6, .8]]))
        self.assertTrue(torch.equal(classified_outputs, torch.Tensor([[0, 0, 1, 1]])))
        evaluated_output = threshold.evaluate(torch.tensor([[0.1, 0.2, .6, .8]]), torch.tensor([[0., 0., 0., 1.]]))
        self.assertEqual(evaluated_output, torch.Tensor([0]))

    def test_score_threshold_trained(self):
        threshold = ScoreThreshold()
        threshold.train(torch.tensor([[0.1, 0.2, .6, .8]]), torch.tensor([[0., 0., 0., 1.]]))
        self.assertAlmostEqual(threshold.midpoint, 0.7)
        classified_outputs = threshold.classify_outputs(torch.tensor([[0.1, 0.2, .6, .8]]))
        self.assertTrue(torch.equal(classified_outputs, torch.Tensor([[0, 0, 0, 1]])))
        evaluated_output = threshold.evaluate(torch.tensor([[0.1, 0.2, .6, .8]]), torch.tensor([[0., 0., 0., 1.]]))
        self.assertEqual(evaluated_output, torch.tensor([1]))

