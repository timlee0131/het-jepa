import torch
import torch_geometric.transforms as T
# from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load(dataset_name, data_dir, pe_k):
    pre_transforms = T.Compose([T.AddLaplacianEigenvectorPE(k=pe_k, attr_name='laplacian_eigenvector_pe', is_undirected=True)])
    dataset = DataLoader(Planetoid(root=data_dir, name=dataset_name, transform=pre_transforms), batch_size=1, shuffle=False)
    return dataset

def load_no_loader(dataset_name, data_dir, pe_k):
    dataset = Planetoid(root=data_dir, name=dataset_name)[0]
    transform = T.AddLaplacianEigenvectorPE(k=pe_k)
    data = transform(dataset)
    return data