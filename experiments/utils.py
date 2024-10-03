import torch
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.data import Data

def data_preprocess(config, data):
    transform = AddLaplacianEigenvectorPE(k=config.pe_k)
    data = transform(data)
    
    sampling_percentage = config.target_percentage
    num_target_nodes = int(data.num_nodes * sampling_percentage)
    target_nodes = torch.randint(0, data.num_nodes, (num_target_nodes,))

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