class BERTEnhancedGraphNetwork(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', bert_dim=768,
                 hidden_dim=300, num_gcn_layers=4, alpha=0.75, beta=0.12,
                 lstm_dropout=0.3, gcn_dropout=0.5):
        super().__init__()

        config = BertConfig.from_pretrained(bert_model_name)
        config.hidden_size = bert_dim
        self.bert = BertModel.from_pretrained(bert_model_name, config=config,
                                              output_hidden_states=True,
                                              output_attentions=True)

        self.bilstm = BiLSTM(bert_dim, hidden_dim, hidden_dim, dropout=lstm_dropout)

        self.bert_dim = bert_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.alpha = alpha
        self.beta = beta

        self.bert_layer_indices = [0, 4, 8, 11]

        self.bert_projection = nn.Linear(bert_dim, hidden_dim)

        self.gcn_layers = nn.ModuleList([
            GraphConvolutionLayer(hidden_dim, hidden_dim)
            for _ in range(num_gcn_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gcn_layers)
        ])

        self.dropout = nn.Dropout(gcn_dropout)

    def extract_attention_adj_matrix(self, attentions, adj_matrix, layer_idx):
        attention = attentions[self.bert_layer_indices[layer_idx]]
        avg_attention = attention.mean(dim=1)

        batch_size, seq_len = avg_attention.shape[0], avg_attention.shape[1]
        sup_adj = torch.zeros_like(avg_attention)

        if adj_matrix is not None:
            adj_seq_len = min(adj_matrix.shape[-1], seq_len)
            for b in range(batch_size):
                for i in range(adj_seq_len):
                    for j in range(adj_seq_len):
                        if avg_attention[b, i, j] > self.alpha:
                            sup_adj[b, i, j] = 1
                        elif avg_attention[b, i, j] < self.beta:
                            sup_adj[b, i, j] = 0
                        else:
                            if b < adj_matrix.shape[0]:
                                sup_adj[b, i, j] = adj_matrix[b, i, j]

                for i in range(adj_seq_len, seq_len):
                    sup_adj[b, i, i] = 1
        else:
            sup_adj = (avg_attention > self.alpha).float()

        return sup_adj

    def forward(self, input_ids, attention_mask, adj_matrix=None, aspect_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        batch_size, seq_len = input_ids.shape

        if adj_matrix is not None and adj_matrix.shape[-1] != seq_len:
            current_size = adj_matrix.shape[-1]

            if current_size < seq_len:
                pad_size = seq_len - current_size
                adj_matrix_padded = torch.zeros(batch_size, seq_len, seq_len,
                                                device=adj_matrix.device)
                adj_matrix_padded[:, :current_size, :current_size] = adj_matrix
                for i in range(current_size, seq_len):
                    adj_matrix_padded[:, i, i] = 1.0
                adj_matrix = adj_matrix_padded
            else:
                adj_matrix = adj_matrix[:, :seq_len, :seq_len]

        if aspect_mask is not None and aspect_mask.shape[-1] != seq_len:
            current_size = aspect_mask.shape[-1]

            if current_size < seq_len:
                aspect_mask_padded = torch.zeros(batch_size, seq_len,
                                                 device=aspect_mask.device)
                aspect_mask_padded[:, :current_size] = aspect_mask
                aspect_mask = aspect_mask_padded
            else:
                aspect_mask = aspect_mask[:, :seq_len]

        bert_last_hidden = hidden_states[-1]
        lstm_output = self.bilstm(bert_last_hidden, attention_mask)

        gcn_inputs = []
        for idx in self.bert_layer_indices:
            bert_layer_output = hidden_states[idx + 1]
            projected = self.bert_projection(bert_layer_output)
            gcn_inputs.append(projected)

        gcn_outputs = []
        for layer_idx in range(self.num_gcn_layers):
            if layer_idx == 0:
                layer_input = gcn_inputs[layer_idx]
            else:
                layer_input = gcn_outputs[-1] + gcn_inputs[layer_idx]

            sup_adj = self.extract_attention_adj_matrix(attentions, adj_matrix, layer_idx)

            gcn_output = self.gcn_layers[layer_idx](layer_input, sup_adj)
            gcn_output = self.layer_norms[layer_idx](gcn_output)
            gcn_output = F.relu(gcn_output)
            gcn_output = self.dropout(gcn_output)

            gcn_outputs.append(gcn_output)

        final_output = gcn_outputs[-1]
        final_output = final_output + lstm_output

        if aspect_mask is not None and aspect_mask.dim() == 2:
            aspect_mask_expanded = aspect_mask.unsqueeze(-1)
            aspect_output = final_output * aspect_mask_expanded
            mask_sum = aspect_mask.sum(dim=1, keepdim=True).clamp(min=1)
            aspect_output = aspect_output.sum(dim=1) / mask_sum
        else:
            aspect_output = final_output.mean(dim=1)

        return aspect_output