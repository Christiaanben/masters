from torch import nn
from torch.optim import Adam
from transformers import DistilBertModel, DistilBertTokenizer


class IntentClassifier(nn.Module):
    def __init__(self, n_labels, optimizer_class=Adam, criterion_class=nn.BCEWithLogitsLoss):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.linear = nn.Linear(768, n_labels)
        self.criterion = criterion_class()
        self.optimizer = optimizer_class(self.parameters(), lr=1e-5)
        self.device = 'cpu'

    def forward(self, inputs):
        tokens = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)
        pretrained_output = self.model(**tokens)
        hidden_state = pretrained_output[0][:, 0]
        output = self.linear(hidden_state)
        return output

    def to(self, device):
        self.device = device
        return super().to(device)
