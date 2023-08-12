import lightning.pytorch as pl
from torch import nn, optim


class IntentPredictor(pl.LightningModule):

    def __init__(self, n_labels: int, criterion_class=nn.BCEWithLogitsLoss):
        super().__init__()
        self.hidden_dim = 80
        self.n_layers = 2
        self.gru = nn.GRU(n_labels, self.hidden_dim, self.n_layers)
        self.fc = nn.Linear(self.hidden_dim, out_features=n_labels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.criterion = criterion_class()

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
        self.log_dict({
            'train_loss': loss,
        },
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=1e-5)
