import json
from abc import ABC, abstractmethod
from typing import List, Dict, Type

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataset import T_co


class Dataset(TorchDataset, ABC):

    def __init__(self, conversations: List[List[Dict]]):
        self.conversations = conversations
        self.intents = self.get_intents()

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

    @abstractmethod
    def __getitem__(self, index):
        """Get item at index from the dataset."""

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the dataset."""


class IntentClassificationDataset(Dataset):

    def __init__(self, conversations: List[List[Dict]]):
        super(IntentClassificationDataset, self).__init__(conversations)
        self.lengths = [len(conversation) for conversation in conversations]
        self.conversation_indices = np.cumsum([0] + self.lengths)

    def _determine_conversation_index(self, index: int) -> int:
        for i in self.conversation_indices[::-1]:
            if i <= index:
                return np.where(self.conversation_indices == i)[0][0]
        return 0

    def __getitem__(self, index: int):
        conversation_index = self._determine_conversation_index(index)
        text_index = index - self.conversation_indices[conversation_index]
        tweet = self.conversations[conversation_index][text_index]
        return tweet.get('text'), self.get_label(tweet.get('intent'))

    def __len__(self) -> int:
        return sum(self.lengths)


class IntentPredictionDataset(Dataset):
    SEQUENCE_LEN = 2  # How many previous messages should be taken into account (for padding and prediction)

    def __init__(self, conversations: List[List[Dict]]):
        super(IntentPredictionDataset, self).__init__(conversations)
        self.lengths = [len(conversation)-1 for conversation in conversations]
        self.conversation_indices = np.cumsum([0] + self.lengths)

    def _determine_conversation_index(self, index: int) -> int:
        for i in self.conversation_indices[::-1]:
            if i <= index:
                return np.where(self.conversation_indices == i)[0][0]
        return 0

    def _get_label_sequence_tensor(self, intents: List[str]):
        stacked_inputs = torch.stack([self.get_label(intent) for intent in intents])
        padded_stacked_inputs = F.pad(stacked_inputs, (0, 0, self.SEQUENCE_LEN, 0))[-2:]
        return padded_stacked_inputs

    def __getitem__(self, index: int):
        conversation_index = self._determine_conversation_index(index)
        text_index = index - self.conversation_indices[conversation_index] + 1
        conversation = self.conversations[conversation_index]
        intents = [tweet.get('intent') for tweet in conversation[:text_index+1]]
        inputs = self._get_label_sequence_tensor(intents[:-1])
        targets = self.get_label(intents[-1])
        return inputs, targets

    def __len__(self) -> int:
        return sum(self.lengths)


if __name__ == '__main__':
    DATA_FILE_NAME = '../data/clean/customer_support_twitter_sample.json'
    data = json.load(open(DATA_FILE_NAME, 'r'))
    dataset = IntentPredictionDataset(data)
    print(dataset.conversations)
    print(len(dataset))
    print(dataset[0])
