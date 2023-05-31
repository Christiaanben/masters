from typing import List, Dict

import numpy as np
import torch

from .dataset import Dataset


class IntentClassificationDataset(Dataset):

    def __init__(self, conversations: List[List[Dict]], intents=None, tokenizer=None):
        super(IntentClassificationDataset, self).__init__(conversations, intents=intents)
        self.lengths = [len(conversation) for conversation in conversations]
        self.conversation_indices = np.cumsum([0] + self.lengths)
        self.tokenizer = tokenizer

    def _determine_conversation_index(self, index: int) -> int:
        for i in self.conversation_indices[::-1]:
            if i <= index:
                return np.where(self.conversation_indices == i)[0][0]
        return 0

    def __getitem__(self, index: int):
        conversation_index = self._determine_conversation_index(index)
        text_index = index - self.conversation_indices[conversation_index]
        tweet = self.conversations[conversation_index][text_index]
        inputs = self.tokenizer(tweet.get('text'), padding="max_length", truncation=True, max_length=512, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(self.get_label(tweet.get('intents')))
        }

    def __len__(self) -> int:
        return sum(self.lengths)
