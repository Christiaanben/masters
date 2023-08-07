from typing import Optional

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from transformers import DistilBertForSequenceClassification
import lightning.pytorch as pl
import torchmetrics


class IntentClassifier(pl.LightningModule):
    model_name = 'distilbert-base-uncased'

    def __init__(self, n_labels):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=n_labels,
            problem_type='multi_label_classification',
            return_dict=True
        )
        self.accuracy = torchmetrics.Accuracy(
            task='multilabel',
            num_labels=n_labels,
        )
        self.f1_score = torchmetrics.F1Score(
            task='multilabel',
            num_labels=n_labels,
            average='macro',
        )

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        accuracy = self.accuracy(outputs.logits, batch['labels'])
        # f1_score = self.f1_score(outputs.logits, batch['labels'])
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            # 'train_f1_score': f1_score,
        },
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=1e-5)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        outputs = self(batch)
        accuracy = self.accuracy(outputs.logits, batch['labels'])
        f1_score = self.f1_score(outputs.logits, batch['labels'])
        self.log_dict({
            'val_loss': outputs.loss,
            'val_accuracy': accuracy,
            'val_f1_score': f1_score,
        },
            prog_bar=True
        )
        return outputs.loss.item()
