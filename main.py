def main():
    config = {
        'data_dir': 'E:\\data\\work\\',
        'bert_model_name': 'bert-base-uncased',
        'batch_size': 32,
        'num_epochs': 30,
        'bert_max_lr': 2e-5,
        'other_max_lr': 1e-3,
        'bert_dim': 768,
        'hidden_dim': 300,
        'num_classes': 3,
        'num_gcn_layers': 4,
        'alpha': 0.75,
        'beta': 0.12,
        'dropout_rate': 0.5,
        'lstm_dropout': 0.3,
        'gcn_dropout': 0.5,
        'max_length': 128,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 60)
    print("SKE Model Training")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Data Directory: {config['data_dir']}")

    tokenizer = BertTokenizer.from_pretrained(config['bert_model_name'])
    gpt4_generator = GPT4KnowledgeGenerator()

    dataset_configs = [
        ('laptop', 'train.json', 'test.json'),
        ('restaurant', 'train_new.json', 'test_new.json'),
    ]

    for domain, train_file, test_file in dataset_configs:
        print(f"\n{'=' * 60}")
        print(f"Training {domain.upper()} Dataset")
        print(f"{'=' * 60}")

        train_path = os.path.join(config['data_dir'], train_file)
        test_path = os.path.join(config['data_dir'], test_file)

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Files not found: {train_path} or {test_path}")
            continue

        train_dataset = ABSADataset(
            data_path=train_path,
            tokenizer=tokenizer,
            max_length=config['max_length'],
            gpt4_generator=gpt4_generator
        )

        test_dataset = ABSADataset(
            data_path=test_path,
            tokenizer=tokenizer,
            max_length=config['max_length'],
            gpt4_generator=gpt4_generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if config['device'] == 'cuda' else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if config['device'] == 'cuda' else False
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        model = SKEModel(
            bert_model_name=config['bert_model_name'],
            bert_dim=config['bert_dim'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes'],
            num_gcn_layers=config['num_gcn_layers'],
            alpha=config['alpha'],
            beta=config['beta'],
            dropout_rate=config['dropout_rate'],
            tokenizer=tokenizer,
            lstm_dropout=config['lstm_dropout'],
            gcn_dropout=config['gcn_dropout']
        )

        trainer = SKETrainer(
            model=model,
            tokenizer=tokenizer,
            gpt4_generator=gpt4_generator,
            device=config['device'],
            bert_max_lr=config['bert_max_lr'],
            other_max_lr=config['other_max_lr']
        )

        total_steps = len(train_loader) * config['num_epochs']
        trainer.set_learning_rate_schedule(total_steps)

        best_accuracy = 0
        best_macro_f1 = 0
        best_epoch = 0

        for epoch in range(1, config['num_epochs'] + 1):
            start_time = time.time()

            train_loss, train_acc = trainer.train_epoch(train_loader, epoch)

            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{config['num_epochs']} - Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            if epoch % 5 == 0 or epoch == config['num_epochs']:
                eval_results = trainer.evaluate(test_loader)
                print(f"Test Loss: {eval_results['loss']:.4f}")
                print(f"Test Acc: {eval_results['accuracy']:.4f}")
                print(f"Test Macro-F1: {eval_results['macro_f1']:.4f}")

                if eval_results['accuracy'] > best_accuracy:
                    best_accuracy = eval_results['accuracy']
                    best_macro_f1 = eval_results['macro_f1']
                    best_epoch = epoch

                    model_save_path = os.path.join(config['data_dir'], f'ske_model_{domain}_best.pt')
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Saved best model: {model_save_path}")

        print(f"\n{domain.upper()} Training Complete!")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"Best Macro-F1: {best_macro_f1:.4f}")

        result_file = os.path.join(config['data_dir'], f'results_{domain}.txt')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"Domain: {domain}\n")
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
            f.write(f"Best Macro-F1: {best_macro_f1:.4f}\n")
            f.write(f"Train Samples: {len(train_dataset)}\n")
            f.write(f"Test Samples: {len(test_dataset)}\n")

    print("\n" + "=" * 60)
    print("All Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()