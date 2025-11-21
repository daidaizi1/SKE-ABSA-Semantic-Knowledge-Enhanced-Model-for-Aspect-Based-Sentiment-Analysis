class BiAffineModule(nn.Module):
    def __init__(self, hidden_dim=300):
        super().__init__()

        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.W4 = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        nn.init.xavier_uniform_(self.U)

    def forward(self, h_sekn, h_begn):
        h_sekn_trans = self.W3(h_sekn)
        h_begn_trans = self.W4(h_begn)

        interaction = torch.matmul(h_sekn_trans, self.U)
        interaction = torch.sum(interaction * h_begn_trans, dim=-1, keepdim=True)

        gate = torch.sigmoid(interaction)

        h_sekn_gated = h_sekn * gate
        h_begn_gated = h_begn * gate

        return h_sekn_gated, h_begn_gated


class SKEModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', bert_dim=768,
                 hidden_dim=300, num_classes=3, num_gcn_layers=4,
                 alpha=0.75, beta=0.12, dropout_rate=0.5, tokenizer=None,
                 lstm_dropout=0.3, gcn_dropout=0.5):
        super().__init__()

        self.tokenizer = tokenizer

        self.begn = BERTEnhancedGraphNetwork(
            bert_model_name=bert_model_name,
            bert_dim=bert_dim,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gcn_layers,
            alpha=alpha,
            beta=beta,
            lstm_dropout=lstm_dropout,
            gcn_dropout=gcn_dropout
        )

        self.sekn = SemanticEnhancedKnowledgeNetwork(
            bert_model_name=bert_model_name,
            bert_dim=bert_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            tokenizer=tokenizer
        )

        self.biaffine = BiAffineModule(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, batch_data):
        h_begn = self.begn(
            input_ids=batch_data['begn_input']['input_ids'],
            attention_mask=batch_data['begn_input']['attention_mask'],
            adj_matrix=batch_data['begn_input']['adj_matrix'],
            aspect_mask=batch_data['begn_input']['aspect_mask']
        )

        h_sekn = self.sekn(
            sentence_ids=batch_data['sekn_input']['sentence_ids'],
            aspect_ids=batch_data['sekn_input']['aspect_ids'],
            expansion_texts=batch_data['sekn_input']['expansion_texts'],
            aspect_knowledge_texts=batch_data['sekn_input']['aspect_knowledge_texts'],
            attention_masks=batch_data['sekn_input']['attention_masks']
        )

        h_sekn_gated, h_begn_gated = self.biaffine(h_sekn, h_begn)

        combined_features = torch.cat([h_sekn_gated, h_begn_gated], dim=-1)

        logits = self.classifier(combined_features)

        return logits