from models.encoders import ContextEncoder, TargetEncoder
from models.evaluator import linear_classifier
from models.predictor import Predictor
from models.mp_jepa import MP_JEPA

from experiments.utils import data_preprocess, sample_and_mask
from experiments.loader import load, load_no_loader

from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
import torch.nn as nn

import importlib
from termcolor import colored, cprint

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

def get_dataset(config):
    dataset = load_no_loader(config.dataset, config.data_dir, config.pe_k)
    
    return dataset

def train(config, data, verbose=False):
    # set up encoders and predictor
    context_encoder = ContextEncoder(config.num_features, config.hidden_channels, config.out_channels)
    target_encoder = TargetEncoder(config.num_features, config.hidden_channels, config.out_channels)
    predictor = Predictor(config.out_channels + config.z_dim * 2, config.out_channels)
    # predictor = Predictor(config.out_channels + config.pe_k, config.out_channels)
    
    model = MP_JEPA(context_encoder, target_encoder, predictor, z_dim=config.z_dim, ema=config.ema)
    
    pos_enc = data.laplacian_eigenvector_pe
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CosineEmbeddingLoss() if config.loss_fn == 'cosine' else nn.MSELoss()
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)
    
    epoch_logger_delta = config.epochs // 10
    epoch_logger_delta += 1 if epoch_logger_delta == 0 else 0
    
    # data = loader[0]
    
    model.train()
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        
        with torch.no_grad():
            target_embedding = model.target_encoder(data.x, data.edge_index)
        
        # for data in loader:
        # loss = 0
        for i in range(config.accumulations):
            target_nodes, masked_x = sample_and_mask(data.x, config.target_percentage)
            
            context_embedding = model.context_encoder(masked_x, data.edge_index)
            
            loss = model(data, context_embedding, target_embedding, target_nodes, criterion)
            
            loss.backward()
            optimizer.step()
    
        # loss = 0
        # for i, batch in enumerate(pred):
        #     # batch_preds = torch.stack(batch)
        #     target_embedding = target_embeddings[i].unsqueeze(0).repeat(batch.size(0), 1)
            
        #     # batch_loss = criterion(batch, target_embedding.detach())
        #     loss += (batch_loss / (len(batch)))
        optimizer.zero_grad()
        lr_scheduler.step()
        
        if verbose:
            if epoch % epoch_logger_delta == 0:
                epoch_c = colored(epoch, 'yellow')
                loss_c = colored(round(loss.item(), 3), 'red')
                print(f'Epoch: {epoch_c}, Loss: {loss_c}')
            
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