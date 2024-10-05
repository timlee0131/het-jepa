from models.encoders import ContextEncoder, TargetEncoder
from models.evaluator import linear_classifier
from models.predictor import Predictor
from models.mp_jepa import MP_JEPA
from experiments.utils import data_preprocess

from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid

import importlib
from termcolor import colored, cprint

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

def train(config, data, verbose=False):
    # set up encoders and predictor
    context_encoder = ContextEncoder(config.num_features, config.hidden_channels, config.out_channels)
    target_encoder = TargetEncoder(config.num_features, config.hidden_channels, config.out_channels)
    predictor = Predictor(config.out_channels + config.z_dim + config.pe_k * 2, config.out_channels)
    # predictor = Predictor(config.out_channels + config.pe_k, config.out_channels)
    
    model = MP_JEPA(context_encoder, target_encoder, predictor, z_dim=config.z_dim, ema=config.ema)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CosineEmbeddingLoss() if config.loss_fn == 'cosine' else nn.MSELoss()
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)
    
    epoch_logger_delta = config.epochs // 10
    epoch_logger_delta += 1 if epoch_logger_delta == 0 else 0
    
    model.train()
    for epoch in range(config.epochs):
        if epoch % 50 == 0:
            data, masked_data, target_nodes = data_preprocess(config, data)
        
        model.train()
        optimizer.zero_grad()
        
        pred, target_embeddings = model(data, masked_data, data.edge_index, target_nodes)
        
        loss = 0
        target_index = 0
        for batch in pred:
            batch_loss = 0
            for pred_i in batch:
                batch_loss += criterion(pred_i, target_embeddings[target_index].unsqueeze(0).detach())
            
            batch_loss /= len(batch)
            loss += batch_loss
            target_index += 1
        
        if verbose:
            if epoch % epoch_logger_delta == 0:
                epoch_c = colored(epoch, 'blue')
                loss_c = colored(loss.item(), 'yellow')
                print(f'Epoch: {epoch_c}, Loss: {loss_c}')
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        model.update_target_encoder()
    
    return model

def driver(config_name):
    config_path = f'./experiments/configs/{config_name}.py'
    
    config = get_config(config_path)
    data = get_dataset(config)
    
    model = train(config, data, verbose=True)
    with torch.no_grad():
        pretrained_representations = model.target_encoder(data.x, data.edge_index)
        
    linear_classifier(config, pretrained_representations, data, verbose=True)
    
    # print base performance
    linear_classifier(config, data.x, data, verbose=True)