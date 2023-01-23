from typing import List, Dict

import numpy as np

from .dataset import Dataset


class IntentClassificationDataset(Dataset):

    def __init__(self, conversations: List[List[Dict]], intents=None):
        super(IntentClassificationDataset, self).__init__(conversations, intents=intents)
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
