import lightning.pytorch as pl
import torchmetrics
from torch import nn, optim


class IntentPredictor(pl.LightningModule):

    hidden_dim = 80
    n_gru_layers = 2
    criterion_class = nn.BCEWithLogitsLoss

    def __init__(self, n_labels: int):
        super().__init__()
        self.gru = nn.GRU(n_labels, self.hidden_dim, self.n_gru_layers)
        self.fc = nn.Linear(self.hidden_dim, n_labels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.criterion = self.criterion_class()
        self.accuracy = torchmetrics.Accuracy(
            task='multilabel',
            num_labels=n_labels,
        )
        self.f1_score = torchmetrics.F1Score(
            task='multilabel',
            num_labels=n_labels,
            average='micro',
        )

    def forward(self, inputs):
        # GRU layers
        output, _ = self.gru(inputs)

        # We're interested in the last output for prediction,
        # which is the contextually richest. If x has shape (batch_size, seq_len, input_dim),
        # out will have shape (batch_size, seq_len, hidden_dim).
        # Thus, we select out[:,-1,:]
        output = output[:, -1, :]

        # Output layer
        output = self.relu(output)
        output = self.fc(output)
        # output = self.sigmoid(output)
        return output

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        accuracy = self.accuracy(outputs, targets)
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
        },
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        accuracy = self.accuracy(outputs, targets)
        f1_score = self.f1_score(outputs, targets)
        self.log_dict({
            'val_loss': loss,
            'val_accuracy': accuracy,
            'val_f1_score': f1_score,
        },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=1e-5)
