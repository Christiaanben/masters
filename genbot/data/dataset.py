from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset, ABC):

    def __init__(self, conversations: List[List[Dict]], intents=None):
        self.conversations = conversations
        self.intents = intents or self.get_intents()

    def get_intents(self) -> List[str]:
        intents = set()
        for conversation in self.conversations:
            for tweet in conversation:
                for intent in tweet.get('intent').split('+'):
                    intents.add(intent)
        return list(intents)

    def get_label(self, intents):
        label = torch.zeros(len(self.intents))
        for intent in intents.split('+'):
            label[self.intents.index(intent)] = 1
        return label

    @property
    def n_labels(self):
        return len(self.intents)

    @abstractmethod
    def __getitem__(self, index):
        """Get item at index from the dataset."""

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the dataset."""
