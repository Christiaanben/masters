from torch import nn, optim
import lightning.pytorch as pl


class IntentPredictor(pl.LightningModule):

    def __init__(self, n_labels: int, optimizer_class=optim.Adam, criterion_class=nn.BCEWithLogitsLoss):
        super().__init__()
        self.hidden_dim = 80
        self.n_layers = 2
        self.rnn = nn.GRU(n_labels, self.hidden_dim, self.n_layers)
        self.fc = nn.Linear(self.hidden_dim, out_features=n_labels)
        self.relu = nn.ReLU()
        self.h0 = self.init_hidden(2)
        self.optimizer = optimizer_class(self.parameters(), lr=1e-5)
        self.criterion = criterion_class()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden

    def forward(self, inputs):
        self.h0 = self.h0.to(inputs.device)
        output, hn = self.rnn(inputs, self.h0)
        output = self.fc(self.relu(output[:, -1]))
        return output

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=1e-5)
