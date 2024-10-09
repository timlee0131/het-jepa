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
    config.dataset = 'Cora'
    config.num_features = 1433
    config.num_classes = 7
    
    # creating the model
    config.loss_fn = 'mse'
    config.hidden_channels = 10124
    config.out_channels = 1024
    config.z_dim = 64
    config.ema = 0.999
    config.target_percentage = 0.1
    config.pe_k = 8
    
    # training settings
    config.runs = 1
    config.epochs = 500
    config.lr = 0.001
    config.min_lr = 0.0001
    config.accumulations = 10
    
    # eval settings
    config.eval_runs = 10
    config.eval_lr = 0.001
    config.eval_epochs = 10
    
    return config