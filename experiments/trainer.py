from models.encoders import ContextEncoder, TargetEncoder
from models.evaluator import linear_classifier
from models.predictor import Predictor
from models.mp_jepa import MP_JEPA

from experiments.utils import data_preprocess, sample_and_mask
from experiments.loader import load, load_no_loader

from torch.optim.lr_scheduler import CosineAnnealingLR
from experiments.schedulers import CosineDecayScheduler, CosineDecayWithRestartsScheduler

import torch
import torch.nn as nn

import importlib
from termcolor import colored, cprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

def get_dataset(config):
    dataset = load_no_loader(config.dataset, config.data_dir, config.pe_k).to(device)
    
    return dataset

def get_schedulers(config):
    lr_scheduler = CosineDecayWithRestartsScheduler(
        max_val=config.lr,
        min_val=config.min_lr,
        warmup_steps=config.warmup_steps,
        total_steps=config.epochs,
        restart_period=config.restart_period
    )
    
    wd_scheduler = CosineDecayWithRestartsScheduler(
        max_val=config.weight_decay[0],
        min_val=1e-5,
        warmup_steps=config.warmup_steps,
        total_steps=config.epochs,
        restart_period=config.restart_period,
    )
    ema_scheduler = CosineDecayWithRestartsScheduler(
        max_val=1 - config.ema,
        min_val=0.0,
        warmup_steps=config.warmup_steps,
        total_steps=config.epochs,
        restart_period=config.epochs
    )
    
    return lr_scheduler, wd_scheduler, ema_scheduler

def train(config, data, verbose=False):
    # set up encoders and predictor
    context_encoder = ContextEncoder(config.num_features, config.hidden_channels, config.out_channels).to(device)
    target_encoder = TargetEncoder(config.num_features, config.hidden_channels, config.out_channels).to(device)
    predictor = Predictor(config.out_channels + config.z_dim * 2, config.out_channels).to(device)
    # predictor = Predictor(config.out_channels + config.pe_k, config.out_channels)
    
    model = MP_JEPA(context_encoder, target_encoder, predictor, z_dim=config.z_dim).to(device)
    
    pos_enc = data.laplacian_eigenvector_pe
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CosineEmbeddingLoss() if config.loss_fn == 'cosine' else nn.MSELoss()
    lr_scheduler, wd_scheduler, ema_scheduler = get_schedulers(config)
    
    def update_lr(step):
        new_lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        new_wd = wd_scheduler.get(step)
        for param_group in optimizer.param_groups:
            if "WD_exclude" not in param_group:
                param_group["weight_decay"] = new_wd
        return new_lr, new_wd
    
    epoch_logger_delta = config.epochs // 10
    epoch_logger_delta += 1 if epoch_logger_delta == 0 else 0
    
    # data = loader[0]
    
    model.train()
    for epoch in range(config.epochs):
        model.train()
        
        new_lr, new_wd = update_lr(epoch)
        optimizer.zero_grad()
        
        with torch.no_grad():
            target_embedding = model.target_encoder(data.x, data.edge_index)
        
        for i in range(config.accumulations):
            target_nodes, masked_x = sample_and_mask(data.x, config.target_percentage)
            context_embedding = model.context_encoder(masked_x, data.edge_index)
            
            loss = model(data, context_embedding, target_embedding, target_nodes, criterion)
            loss.backward()
            
        optimizer.step()
        optimizer.zero_grad()
        # lr_scheduler.step() -- from the torch scheduler, Eli's scheduler overrides this
        
        new_ema = 1 - ema_scheduler.get(epoch)
        model.update_target_encoder(new_ema)
        
        if verbose:
            if epoch % epoch_logger_delta == 0:
                epoch_c = colored(epoch, 'yellow')
                loss_c = colored(round(loss.item(), 3), 'red')
                print(f'Epoch: {epoch_c}, Loss: {loss_c}')
    
    return model

def driver(config_name):
    config_path = f'./experiments/configs/{config_name}.py'
    
    config = get_config(config_path)
    data = get_dataset(config)
    
    model = train(config, data, verbose=True)
    with torch.no_grad():
        pretrained_representations = model.target_encoder(data.x, data.edge_index)
        
    linear_classifier(config, pretrained_representations.cpu(), data.cpu(), verbose=True)
    
    # print base performance
    linear_classifier(config, data.x.cpu(), data.cpu(), base=True, verbose=True)