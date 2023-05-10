from transformers import GPT2Tokenizer

from .dataset import Dataset


class GeneratorDataset(Dataset):

    def __init__(self, tokenizer: GPT2Tokenizer):
        super().__init__([[]])
        self.tokenizer = tokenizer

        input_texts = ['yo', 'how are you?', 'what is your name?', 'what is your favorite color?']
        expected_outputs = ['hey', "I'm doing well, thanks. How about you?", "My name is DialoGPT.", 'blue']
        data = []
        for in_text, out_text in zip(input_texts, expected_outputs):
            data.append(f'{in_text}{self.tokenizer.eos_token}{out_text}{self.tokenizer.eos_token}')
        self.tokens = self.tokenizer(data, padding=True, truncation=True, return_tensors='pt')

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens['input_ids'][idx], self.tokens['attention_mask'][idx]
