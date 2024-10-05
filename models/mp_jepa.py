import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

class MP_JEPA(nn.Module):
    def __init__(self, context_encoder, target_encoder, predictor, z_dim, ema=0.999):
        super(MP_JEPA, self).__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        
        self.z = nn.Parameter(torch.Tensor(1, z_dim))
        nn.init.xavier_uniform_(self.z)
        self.pe_projector = nn.Linear(4, 4, bias=False)
        
        self.z_dim = z_dim
        self.ema = ema

    def forward(self, data, masked_data, edge_index, target_nodes):
        x, masked_x = data.x, masked_data.x
        pos_enc = data.laplacian_eigenvector_pe
        
        with torch.no_grad():
            target_x = self.target_encoder(x, edge_index)

        target_x = target_x.detach()
        target_embeddings = target_x[target_nodes]
        # target_embeddings = torch.cat([target_embeddings, pos_enc[target_nodes]], dim=1)
        
        total_context_embedding = self.context_encoder(masked_x, edge_index)
        
        context_embeddings = []
        total_pred = []
        for node in target_nodes:
            sub_nodes, sub_edge_index, _,_ = k_hop_subgraph(int(node), num_hops=1, edge_index=edge_index, relabel_nodes=True)
            
            subgraph_x = masked_x[sub_nodes]
            subgraph_edge_index = sub_edge_index
            
            # context_embedding = self.context_encoder(subgraph_x, subgraph_edge_index)
            context_embedding = total_context_embedding[sub_nodes]
            
            position_encoding = torch.cat([self.pe_projector(pos_enc[sub_nodes]), self.pe_projector(pos_enc[node].unsqueeze(0)).repeat(context_embedding.size(0), 1)], dim=1)
            # print(context_embedding.size(), self.z.repeat(context_embedding.size(0), 1).size(), position_encoding.size())
            context_embedding = torch.cat([context_embedding, self.z.repeat(context_embedding.size(0), 1), position_encoding], dim=1)
            batch_pred = []
            for i in range(len(sub_nodes)):
                batch_pred.append(self.predictor(context_embedding[i].unsqueeze(0)))
            
            total_pred.append(batch_pred)
        
        # context_embeddings = torch.stack(context_embeddings, dim=0)
        # total_pred = torch.stack(total_pred, dim=0)
        # predictor
        return total_pred, target_embeddings
    
    def cosine_similarity(self, x, y):
        return 1 - F.cosine_similarity(x, y, dim=-1).mean()
        
    @torch.no_grad()
    def update_target_encoder(self):
        for context_params, target_params in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = target_params.data * self.ema + context_params.data * (1 - self.ema)