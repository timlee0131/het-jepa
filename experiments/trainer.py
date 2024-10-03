from models.encoders import ContextEncoder, TargetEncoder
from models.evaluator import linear_classifier
from models.predictor import Predictor
from models.mp_jepa import MP_JEPA
from experiments.utils import data_preprocess

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
    data, masked_data, target_nodes = data_preprocess(config, data)
    
    # set up encoders and predictor
    context_encoder = ContextEncoder(config.num_features, config.hidden_channels, config.out_channels)
    target_encoder = TargetEncoder(config.num_features, config.hidden_channels, config.out_channels)
    predictor = Predictor(config.out_channels, config.num_features, config.z_dim)
    
    model = MP_JEPA(context_encoder, target_encoder, predictor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CosineEmbeddingLoss() if config.loss_fn == 'cosine' else nn.MSELoss()
    
    epoch_logger_delta = config.epochs // 10
    
    model.train()
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        
        pred, target_embeddings = model(data, masked_data, data.edge_index, target_nodes)
        
        loss = criterion(pred, target_embeddings.detach())
        
        if verbose:
            if epoch % epoch_logger_delta == 0:
                epoch_c = colored(epoch, 'blue')
                loss_c = colored(loss.item(), 'magenta')
                print(f'Epoch: {epoch_c}, Loss: {loss_c}')
        
        loss.backward()
        optimizer.step()
        
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
    