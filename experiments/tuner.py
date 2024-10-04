from models.encoders import ContextEncoder, TargetEncoder
from models.evaluator import linear_classifier, linear_classifier_custom
from models.predictor import Predictor
from models.mp_jepa import MP_JEPA
from experiments.utils import data_preprocess

from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid

import importlib
from termcolor import colored, cprint

import optuna

def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

def get_dataset(config):
    dataset = None
    if config.dataset == 'Cora':
        dataset = Planetoid(root=config.data_dir, name='Cora')
        dataset = dataset[0]
    elif config.dataset == 'CiteSeer':
        dataset = Planetoid(root=config.data_dir, name='CiteSeer')
        dataset = dataset[0]
    elif config.dataset == 'PubMed':
        dataset = Planetoid(root=config.data_dir, name="PubMed")
        dataset = dataset[0]
    else:
        cprint("invalid dataset...", "red")
    
    return dataset

def train(config, params, data, verbose=False):
    data, masked_data, target_nodes = data_preprocess(config, data)
    
    # set up encoders and predictor
    context_encoder = ContextEncoder(config.num_features, params['hidden_channels'], params['hidden_channels'])
    target_encoder = TargetEncoder(config.num_features, params['hidden_channels'], params['hidden_channels'])
    predictor = Predictor(params['hidden_channels'], config.num_features, params['z_dim'])
    
    model = MP_JEPA(context_encoder, target_encoder, predictor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    criterion = nn.CosineEmbeddingLoss() if config.loss_fn == 'cosine' else nn.MSELoss()
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=params['epochs'], eta_min=config.min_lr)
    
    epoch_logger_delta = params['epochs'] // 10
    
    model.train()
    for epoch in range(params['epochs']):
        model.train()
        optimizer.zero_grad()
        
        pred, target_embeddings = model(data, masked_data, data.edge_index, target_nodes)
        # print(pred.shape, target_embeddings.shape)
        
        loss = criterion(pred, target_embeddings.detach())
        
        if verbose:
            if epoch % epoch_logger_delta == 0:
                epoch_c = colored(epoch, 'blue')
                loss_c = colored(loss.item(), 'magenta')
                print(f'Epoch: {epoch_c}, Loss: {loss_c}')
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        model.update_target_encoder()
    
    return model

def tuning(trial: optuna.Trial, config, data):
    params = {
        'hidden_channels': trial.suggest_categorical('hidden_channels', config.hidden_channels),
        'z_dim': trial.suggest_categorical('z_dim', config.z_dim),
        'epochs': trial.suggest_categorical('epochs', config.epochs)
    }
    
    model = train(config, params, data)
    
    with torch.no_grad():
        pretrained_representations = model.target_encoder(data.x, data.edge_index)
        
    return linear_classifier(config, pretrained_representations, data)

def driver(config_name):
    config_path = f'./experiments/configs/tuners/{config_name}.py'
    
    config = get_config(config_path)
    data = get_dataset(config)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: tuning(trial, config, data), n_trials=config.n_optuna)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")