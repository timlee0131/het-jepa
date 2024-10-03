import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

class MP_JEPA(nn.Module):
    def __init__(self, context_encoder, target_encoder, predictor, ema=0.99):
        super(MP_JEPA, self).__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        
        self.ema = ema

    def forward(self, data, masked_data, edge_index, target_nodes):
        x, masked_x = data.x, masked_data.x
        pos_enc = data.laplacian_eigenvector_pe
        
        with torch.no_grad():
            target_x = self.target_encoder(x, edge_index)

        target_x = target_x.detach()
        target_embeddings = target_x[target_nodes]
        target_embeddings = torch.cat([target_embeddings, pos_enc[target_nodes]], dim=1)
        
        context_embeddings = []
        for node in target_nodes:
            sub_nodes, sub_edge_index, _,_ = k_hop_subgraph(int(node), num_hops=2, edge_index=edge_index, relabel_nodes=True)
            
            subgraph_x = masked_x[sub_nodes]
            # subgraph_x[node] = random_feature_vector[0]
            subgraph_edge_index = sub_edge_index
            
            context_embedding = self.context_encoder(subgraph_x, subgraph_edge_index)
            context_embedding = torch.cat([context_embedding, pos_enc[sub_nodes]], dim=1)
            # context_embedding = context_embedding + pos_enc[sub_nodes]
            context_embeddings.append(context_embedding.mean(dim=0))    # mean pooling
        
        context_embeddings = torch.stack(context_embeddings, dim=0)
        
        # predictor
        pred = self.predictor(context_embeddings)
        return pred, target_embeddings
    
    def cosine_similarity(self, x, y):
        return 1 - F.cosine_similarity(x, y, dim=-1).mean()
        
    @torch.no_grad()
    def update_target_encoder(self):
        for context_params, target_params in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = target_params.data * self.ema + context_params.data * (1 - self.ema)