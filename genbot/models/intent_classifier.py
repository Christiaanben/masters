from torch import nn
from torch.optim import Adam
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


class IntentClassifier(nn.Module):
    model_name = 'distilbert-base-uncased'

    def __init__(self, n_labels, optimizer_class=Adam, criterion_class=nn.BCEWithLogitsLoss):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=n_labels,
            problem_type='multi_label_classification',
            return_dict=True
        )
        self.linear = nn.Linear(768, n_labels)
        self.criterion = criterion_class()
        self.optimizer = optimizer_class(self.parameters(), lr=1e-5)
        self.device = 'cpu'

    @classmethod
    def init_tokenizer(cls) -> DistilBertTokenizerFast:
        return DistilBertTokenizerFast.from_pretrained(cls.model_name)

    def forward(self, inputs):
        return self.model(**inputs)

    def to(self, device):
        self.device = device
        return super().to(device)
