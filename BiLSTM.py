class BiLSTM(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=300, output_dim=300, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.output_projection = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        lstm_out, _ = self.lstm(x)
        output = self.output_projection(lstm_out)
        output = self.dropout(output)
        return output


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input_features, adj_matrix):
        support = torch.matmul(input_features, self.weight)
        output = torch.matmul(adj_matrix, support)
        degree = adj_matrix.sum(dim=-1, keepdim=True).clamp(min=1)
        output = output / degree
        output = output + self.bias
        return output