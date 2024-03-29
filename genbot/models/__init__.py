from .intent_classifier import IntentClassifier
from .intent_predictor import IntentPredictor
import numpy as np
import torch


class ScoreThreshold:

    def __init__(self):
        self.highs = [1.]
        self.lows = [0.]

    def train(self, outputs, targets):
        acc = self.evaluate(outputs, targets)
        for output, target in zip(outputs, targets):
            for out, tgt in zip(output, target):
                if tgt:
                    self.highs.append(out)
                else:
                    self.lows.append(out)
        self.highs = self.highs[-10:]
        self.lows = self.lows[-10:]
        return acc

    def evaluate(self, outputs, targets):
        return 1 - torch.max(torch.abs(targets - self.classify_outputs(outputs)), axis=1).values

    def classify_outputs(self, outputs):
        return torch.tensor([[1 if v > 0 else 0 for v in (output - self.midpoint)] for output in outputs],
                            dtype=torch.float32)

    @property
    def midpoint(self):
        return np.average([min(self.highs), max(self.lows)])

    def __repr__(self):
        return 'str'

