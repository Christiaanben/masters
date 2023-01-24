from torch import nn


class IntentPredictor(nn.Module):

    def __init__(self):
        super(IntentPredictor, self).__init__()
        self.hidden_dim = 80
        self.n_layers = 2
        self.rnn = nn.GRU(9, self.hidden_dim, self.n_layers)
        self.fc = nn.Linear(self.hidden_dim, 9)
        self.relu = nn.ReLU()
        self.h0 = self.init_hidden(2)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to('cuda')
        return hidden

    def forward(self, inputs):
        output, hn = self.rnn(inputs, self.h0)
        output = self.fc(self.relu(output[:, -1]))
        return output
