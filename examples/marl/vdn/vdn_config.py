VDNConfig = {
    'project': 'StarCraft-II',
    'scenario': '3m',
    'replay_buffer_size': 5000,
    'mixing_embed_dim': 32,
    'rnn_hidden_dim': 64,
    'learning_rate': 0.001,
    'min_learning_rate': 0.00001,
    'memory_warmup_size': 64,
    'gamma': 0.99,
    'exploration_start': 1.0,
    'min_exploration': 0.05,
    'update_target_interval': 100,
    'batch_size': 32,
    'total_episode': 100000,
    'train_log_interval': 10,  # log every 10 episode
    'test_log_interval': 100,  # log every 100 epidode
    'clip_grad_norm': 10,
    'hypernet_layers': 2,
    'hypernet_embed_dim': 64,
    'update_learner_freq': 5,
    'double_q': True,
    'difficulty': '7',
    'algo': 'vdn',
    'log_dir': 'work_dirs/',
    'logger': 'wandb'
}
