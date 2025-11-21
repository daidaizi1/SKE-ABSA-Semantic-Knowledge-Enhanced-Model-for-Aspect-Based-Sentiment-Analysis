import torch

CONFIG = {
    'data_dir': './data/',
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

