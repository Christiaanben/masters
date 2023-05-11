import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn


class Generator(nn.Module):

    pretrained_model = 'microsoft/DialoGPT-small'

    def __init__(self, optimizer_class=AdamW):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.optimizer = optimizer_class(self.parameters(), lr=1e-5)

    @classmethod
    def init_tokenizer(cls):
        tokenizer = AutoTokenizer.from_pretrained(cls.pretrained_model)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def chat(self, n_steps=3):
        # chat for n lines
        for step in range(n_steps):
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            user_text = input(">> User:")
            new_user_input_ids = self.tokenizer.encode(user_text + self.tokenizer.eos_token, return_tensors='pt')
            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids],
                                      dim=-1) if step > 0 else new_user_input_ids
            # generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = self.model.generate(
                bot_input_ids,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id
            )
            # pretty print last ouput tokens from bot
            print("DialoGPT: {}".format(
                    self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
