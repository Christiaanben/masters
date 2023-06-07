from typing import List, Dict

import torch
from torch.nn import functional as F

from .dataset import Dataset


class IntentPredictionDataset(Dataset):
    SEQUENCE_LEN = 2  # How many previous messages should be taken into account (for padding and prediction)

    def __init__(self, conversations: List[List[Dict]]):
        super(IntentPredictionDataset, self).__init__(conversations)
        self.conversation_indices = []
        for conversation_idx, conversation in enumerate(self.conversations):
            for message_idx, message in enumerate(conversation):
                if message.get('authored') and message_idx > 0:
                    self.conversation_indices.append((conversation_idx, message_idx))

    def _get_label_sequence_tensor(self, intents: List[str]):
        stacked_inputs = torch.stack([self.get_label(intent) for intent in intents])
        padded_stacked_inputs = F.pad(stacked_inputs, (0, 0, self.SEQUENCE_LEN, 0))[-2:]
        return padded_stacked_inputs

    def __getitem__(self, index: int):
        conversation_index, text_index = self.conversation_indices[index]
        conversation = self.conversations[conversation_index]
        intents = [tweet.get('intents') for tweet in conversation[:text_index + 1]]
        inputs = self._get_label_sequence_tensor(intents[:-1])
        targets = self.get_label(intents[-1])
        return inputs, targets

    def __len__(self) -> int:
        return len(self.conversation_indices)


if __name__ == '__main__':
    import json

    DATA_FILE_NAME = '../../data/clean/customer_support_twitter_sample.json'
    data = json.load(open(DATA_FILE_NAME, 'r'))
    dataset = IntentPredictionDataset(data)
    print(dataset.conversations)
    print(len(dataset))
    print(dataset[0])
