class SemanticEnhancedKnowledgeNetwork(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', bert_dim=768,
                 hidden_dim=300, dropout_rate=0.5, tokenizer=None):
        super().__init__()

        config = BertConfig.from_pretrained(bert_model_name)
        config.hidden_size = bert_dim
        self.bert = BertModel.from_pretrained(bert_model_name, config=config)

        self.tokenizer = tokenizer
        self.bert_dim = bert_dim
        self.hidden_dim = hidden_dim

        self.projection = nn.Linear(bert_dim, hidden_dim)

        self.sentinel_vector = nn.Parameter(torch.randn(1, hidden_dim))

        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.W_s = nn.Linear(hidden_dim, hidden_dim)

        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def compute_attention(self, query, keys, values, sentinel=True):
        batch_size = query.shape[0]

        query_expanded = query.unsqueeze(1)
        scores = torch.matmul(query_expanded, keys.transpose(-2, -1))
        scores = scores.squeeze(1)

        if sentinel:
            sentinel_expanded = self.sentinel_vector.expand(batch_size, -1)
            sentinel_score = torch.sum(query * sentinel_expanded, dim=-1, keepdim=True)
            scores = torch.cat([scores, sentinel_score], dim=-1)
            values = torch.cat([values, sentinel_expanded.unsqueeze(1)], dim=1)

        attention_weights = F.softmax(scores / math.sqrt(self.hidden_dim), dim=-1)
        attention_weights = attention_weights.unsqueeze(-1)

        weighted_sum = torch.sum(values * attention_weights, dim=1)

        return weighted_sum

    def forward(self, sentence_ids, aspect_ids, expansion_texts=None,
                aspect_knowledge_texts=None, attention_masks=None):
        batch_size = sentence_ids.shape[0]
        device = sentence_ids.device

        sentence_outputs = self.bert(
            input_ids=sentence_ids,
            attention_mask=attention_masks['sentence_mask']
        )
        h_s_bert = sentence_outputs.last_hidden_state[:, 0, :]
        h_s = self.projection(h_s_bert)

        aspect_outputs = self.bert(
            input_ids=aspect_ids,
            attention_mask=attention_masks['aspect_mask']
        )
        h_a_bert = aspect_outputs.last_hidden_state.mean(dim=1)
        h_a = self.projection(h_a_bert)

        if expansion_texts is not None and len(expansion_texts) > 0:
            expansion_features = []
            for exp_text_list in expansion_texts:
                exp_batch_features = []
                for exp_text in exp_text_list[:4]:
                    if self.tokenizer:
                        exp_encoding = self.tokenizer(
                            exp_text,
                            truncation=True,
                            padding='max_length',
                            max_length=128,
                            return_tensors='pt'
                        )
                        exp_ids = exp_encoding['input_ids'].to(device)
                        exp_mask = exp_encoding['attention_mask'].to(device)
                        exp_output = self.bert(input_ids=exp_ids, attention_mask=exp_mask)
                        exp_feature = self.projection(exp_output.last_hidden_state[:, 0, :])
                        exp_batch_features.append(exp_feature)

                if exp_batch_features:
                    exp_batch_features = torch.cat(exp_batch_features, dim=0)
                    expansion_features.append(exp_batch_features)

            if expansion_features:
                max_len = max(f.shape[0] for f in expansion_features)
                padded_features = []
                for f in expansion_features:
                    if f.shape[0] < max_len:
                        padding = torch.zeros(max_len - f.shape[0], f.shape[1], device=device)
                        f = torch.cat([f, padding], dim=0)
                    padded_features.append(f)
                expansion_features = torch.stack(padded_features, dim=0)
            else:
                expansion_features = h_s.unsqueeze(1)
        else:
            expansion_features = h_s.unsqueeze(1)

        if aspect_knowledge_texts is not None and len(aspect_knowledge_texts) > 0:
            knowledge_features = []
            for know_text_list in aspect_knowledge_texts:
                know_batch_features = []
                for know_text in know_text_list[:4]:
                    if self.tokenizer:
                        know_encoding = self.tokenizer(
                            know_text,
                            truncation=True,
                            padding='max_length',
                            max_length=128,
                            return_tensors='pt'
                        )
                        know_ids = know_encoding['input_ids'].to(device)
                        know_mask = know_encoding['attention_mask'].to(device)
                        know_output = self.bert(input_ids=know_ids, attention_mask=know_mask)
                        know_feature = self.projection(know_output.last_hidden_state[:, 0, :])
                        know_batch_features.append(know_feature)

                if know_batch_features:
                    know_batch_features = torch.cat(know_batch_features, dim=0)
                    knowledge_features.append(know_batch_features)

            if knowledge_features:
                max_len = max(f.shape[0] for f in knowledge_features)
                padded_features = []
                for f in knowledge_features:
                    if f.shape[0] < max_len:
                        padding = torch.zeros(max_len - f.shape[0], f.shape[1], device=device)
                        f = torch.cat([f, padding], dim=0)
                    padded_features.append(f)
                knowledge_features = torch.stack(padded_features, dim=0)
            else:
                knowledge_features = h_a.unsqueeze(1)
        else:
            knowledge_features = h_a.unsqueeze(1)

        f_s = self.compute_attention(h_s, expansion_features, expansion_features, sentinel=True)
        f_s = self.layer_norm(f_s + h_s)

        f_a = self.compute_attention(h_a, knowledge_features, knowledge_features, sentinel=True)
        f_a = self.layer_norm(f_a + h_a)

        semantic_enhanced = torch.cat([f_s, f_a], dim=-1)
        semantic_enhanced = self.fusion_layer(semantic_enhanced)
        semantic_enhanced = F.relu(semantic_enhanced)
        semantic_enhanced = self.dropout(semantic_enhanced)

        return semantic_enhanced