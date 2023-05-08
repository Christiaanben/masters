from torch import nn, optim


class IntentPredictor(nn.Module):

    def __init__(self, optimizer_class=optim.Adam, criterion_class=nn.BCEWithLogitsLoss):
        super().__init__()
        self.hidden_dim = 80
        self.n_layers = 2
        self.rnn = nn.GRU(9, self.hidden_dim, self.n_layers)
        self.fc = nn.Linear(self.hidden_dim, 9)
        self.relu = nn.ReLU()
        self.h0 = self.init_hidden(2)
        self.optimizer = optimizer_class(self.parameters(), lr=1e-5)
        self.criterion = criterion_class()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to('cuda')
        return hidden

    def forward(self, inputs):
        output, hn = self.rnn(inputs, self.h0)
        output = self.fc(self.relu(output[:, -1]))
        return output
