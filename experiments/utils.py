import torch
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.data import Data

def sample_and_mask(x, target_percentage):
    tgt_idx = torch.multinomial(
        torch.ones(x.size(0)),
        max(int(target_percentage * x.size(0)), 1),
        replacement=False,
    )

    # replace target nodes with mask token
    mask = torch.zeros(x.size(0), 1)
    mask[tgt_idx] = 1

    # mask is a random vector with the same mean and std as x
    mask_token = torch.randn(1, x.size(-1)) * torch.std(
        x
    ) + torch.mean(x)
    mask_tokens = mask_token.repeat(x.size(0), 1)

    x = x * (1 - mask) + (mask * mask_tokens)

    return tgt_idx, x

def data_preprocess(config, data):
    transform = AddLaplacianEigenvectorPE(k=config.pe_k)
    data = transform(data)
    
    sampling_percentage = config.target_percentage
    num_target_nodes = int(data.num_nodes * sampling_percentage)
    # target_nodes = torch.randint(0, data.num_nodes, (num_target_nodes,))
    probabilities = torch.ones(data.num_nodes)
    target_nodes = torch.multinomial(probabilities, num_target_nodes, replacement=False)

    # generating the random vector centered around mean(total features)
    feature_mean = data.x.mean(dim=0)
    feature_std = data.x.std(dim=0)

    random_feature_vector = torch.randn((num_target_nodes, data.num_features)) * feature_std + feature_mean
    masked_features = data.x.clone()
    # ones_mask = torch.ones(1, data.num_features)
    masked_features[target_nodes] = random_feature_vector

    target_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    target_mask[target_nodes] = True
    data.target_mask = target_mask

    masked_data = Data(
        x = masked_features,
        edge_index = data.edge_index,
        y = data.y,
        train_mask = data.train_mask,
        val_mask = data.val_mask,
        test_mask = data.test_mask,
        target_mask = target_mask
    )
    
    return data, masked_data, target_nodes