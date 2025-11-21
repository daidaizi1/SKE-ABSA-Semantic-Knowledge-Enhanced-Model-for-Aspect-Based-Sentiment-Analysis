class SKETrainer:
    def __init__(self, model, tokenizer, gpt4_generator, device='cuda',
                 bert_max_lr=2e-5, other_max_lr=1e-3):

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.gpt4_generator = gpt4_generator
        self.device = device

        bert_params = []
        other_params = []

        for name, param in model.named_parameters():
            if 'bert' in name:
                bert_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = optim.Adam([
            {'params': bert_params, 'lr': bert_max_lr},
            {'params': other_params, 'lr': other_max_lr}
        ])

        self.criterion = nn.CrossEntropyLoss()

        self.bert_max_lr = bert_max_lr
        self.other_max_lr = other_max_lr

    def set_learning_rate_schedule(self, num_training_steps):
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )

    def prepare_batch_data(self, batch):
        device = self.device

        batch_size = batch['sentence_ids'].shape[0]

        expansion_texts = []
        aspect_knowledge_texts = []

        for i in range(batch_size):
            sentence = batch['sentence'][i] if isinstance(batch['sentence'], list) else batch['sentence']
            aspect = batch['aspect'][i] if isinstance(batch['aspect'], list) else batch['aspect']

            expansions = self.gpt4_generator.generate_sentence_expansion(sentence, beam_size=4)
            expansion_texts.append(expansions)

            knowledge = self.gpt4_generator.generate_aspect_knowledge(sentence, aspect, beam_size=4)
            aspect_knowledge_texts.append(knowledge)

        begn_input = {
            'input_ids': batch['combined_ids'].to(device),
            'attention_mask': batch['combined_mask'].to(device),
            'adj_matrix': batch['adj_matrix'].to(device),
            'aspect_mask': batch['token_aspect_mask'].to(device)
        }

        sekn_input = {
            'sentence_ids': batch['sentence_ids'].to(device),
            'aspect_ids': batch['aspect_ids'].to(device),
            'expansion_texts': expansion_texts,
            'aspect_knowledge_texts': aspect_knowledge_texts,
            'attention_masks': {
                'sentence_mask': batch['sentence_mask'].to(device),
                'aspect_mask': batch['aspect_mask'].to(device)
            }
        }

        return {
            'begn_input': begn_input,
            'sekn_input': sekn_input
        }

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in progress_bar:
            batch_data = self.prepare_batch_data(batch)
            labels = batch['label'].to(self.device)

            logits = self.model(batch_data)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            current_lr = self.optimizer.param_groups[0]['lr']

            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total,
                'lr': current_lr
            })

        return total_loss / len(train_loader), correct / total

    def evaluate(self, eval_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc='Evaluating'):
                batch_data = self.prepare_batch_data(batch)
                labels = batch['label'].to(self.device)

                logits = self.model(batch_data)
                loss = self.criterion(logits, labels)

                _, predicted = torch.max(logits, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()

        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        return {
            'loss': total_loss / len(eval_loader),
            'accuracy': accuracy,
            'macro_f1': macro_f1
        }