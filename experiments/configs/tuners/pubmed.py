import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.computer = "local"
    config.n_gpus = 1 if config.computer == "superpod" else 0
    config.data_dir = (
        "./experiments/data"
        if config.computer == "local"
        else "/data"
    )
    
    # dataset info
    config.dataset = 'PubMed'
    config.num_features = 500
    config.num_classes = 3
    
    # creating the model
    config.loss_fn = 'mse'
    config.hidden_channels = [128, 256, 420]
    config.out_channels = [128, 256, 420]
    config.z_dim = [16, 32, 64, 128]
    config.ema = 0.99
    config.target_percentage = 0.1
    config.pe_k = 4
    
    # training settings
    config.runs = 1
    config.epochs = [100, 200, 500, 1000]
    config.lr = 0.001
    config.min_lr = 0.0001
    
    # eval settings
    config.eval_runs = 1
    config.eval_lr = 0.001
    config.eval_epochs = 10
    
    # tuning parameters
    config.n_optuna = 20
    
    return config