from typing import List, Dict

from transformers import GPT2Tokenizer

from .dataset import Dataset


class GeneratorDataset(Dataset):

    def __init__(self, conversations: List[List[Dict]], tokenizer: GPT2Tokenizer):
        super().__init__(conversations)
        self.tokenizer = tokenizer
        data = []
        for conversation in conversations:
            for input_message, response_message in zip(conversation, conversation[1:]):
                if response_message.get('authored'):
                    data.append(f'{input_message.get("text")}{self.tokenizer.eos_token}{response_message.get("text")}{self.tokenizer.eos_token}')

        self.tokens = self.tokenizer(data, padding=True, truncation=True, return_tensors='pt')

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx):
        inputs = self.tokens['input_ids'][idx]
        labels = self.tokens['input_ids'][idx].clone()
        start_idx = (labels == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
        labels[:start_idx+1] = -100
        return inputs, self.tokens['attention_mask'][idx], labels
