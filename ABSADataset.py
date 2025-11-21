class ABSADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, gpt4_generator=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.gpt4_generator = gpt4_generator

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.samples = []
        self.process_data()

    def process_data(self):
        polarity_map = {'positive': 2, 'neutral': 1, 'negative': 0}

        for item in self.data:
            if 'aspects' not in item or len(item['aspects']) == 0:
                continue

            tokens = item['token']
            dep_heads = item.get('head', [])

            sentence = ' '.join(tokens)

            n = len(tokens)
            adj_matrix = np.zeros((n, n))
            for i, head in enumerate(dep_heads):
                if head > 0 and head <= n:
                    adj_matrix[i][head - 1] = 1
                    adj_matrix[head - 1][i] = 1
                adj_matrix[i][i] = 1

            for aspect_info in item['aspects']:
                aspect_tokens = aspect_info['term']
                aspect_text = ' '.join(aspect_tokens)
                polarity = polarity_map.get(aspect_info['polarity'], 1)
                from_idx = aspect_info['from']
                to_idx = aspect_info['to']

                aspect_mask = np.zeros(n)
                if from_idx < n and to_idx <= n:
                    aspect_mask[from_idx:to_idx] = 1

                self.samples.append({
                    'sentence': sentence,
                    'tokens': tokens,
                    'aspect': aspect_text,
                    'label': polarity,
                    'adj_matrix': adj_matrix,
                    'aspect_mask': aspect_mask,
                    'from_idx': from_idx,
                    'to_idx': to_idx
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        sentence_encoding = self.tokenizer(
            sample['sentence'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        aspect_encoding = self.tokenizer(
            sample['aspect'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        combined_text = f"{sample['sentence']} [SEP] {sample['aspect']}"
        combined_encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        adj_matrix = sample['adj_matrix']
        n = adj_matrix.shape[0]
        if n < self.max_length:
            pad_width = self.max_length - n
            adj_matrix = np.pad(adj_matrix, ((0, pad_width), (0, pad_width)), 'constant')
        else:
            adj_matrix = adj_matrix[:self.max_length, :self.max_length]

        aspect_mask = sample['aspect_mask']
        if len(aspect_mask) < self.max_length:
            aspect_mask = np.pad(aspect_mask, (0, self.max_length - len(aspect_mask)), 'constant')
        else:
            aspect_mask = aspect_mask[:self.max_length]

        token_aspect_mask = np.zeros(self.max_length)

        from_idx = sample['from_idx']
        to_idx = sample['to_idx']

        actual_from = from_idx + 1
        actual_to = to_idx + 1

        if actual_to <= self.max_length:
            token_aspect_mask[actual_from:actual_to] = 1

        return {
            'sentence_ids': sentence_encoding['input_ids'].squeeze(),
            'sentence_mask': sentence_encoding['attention_mask'].squeeze(),
            'aspect_ids': aspect_encoding['input_ids'].squeeze(),
            'aspect_mask': aspect_encoding['attention_mask'].squeeze(),
            'combined_ids': combined_encoding['input_ids'].squeeze(),
            'combined_mask': combined_encoding['attention_mask'].squeeze(),
            'sentence': sample['sentence'],
            'aspect': sample['aspect'],
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'adj_matrix': torch.tensor(adj_matrix, dtype=torch.float),
            'orig_aspect_mask': torch.tensor(aspect_mask, dtype=torch.float),
            'token_aspect_mask': torch.tensor(token_aspect_mask, dtype=torch.float)
        }