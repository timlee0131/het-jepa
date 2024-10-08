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
        self.pe_projector = nn.Linear(4, z_dim, bias=False)
        
        self.z_dim = z_dim
        self.ema = ema

    def forward(self, data, context_embedding, target_embedding, target_nodes, criterion):
        
        position_encoding = self.pe_projector(data.laplacian_eigenvector_pe)
        
        loss = 0
        for node in target_nodes:
            sub_nodes, sub_edge_index, _,_ = k_hop_subgraph(int(node), num_hops=2, edge_index=data.edge_index, relabel_nodes=True)
            
            sub_context_embedding = context_embedding[sub_nodes]

            z_ctx_pe = self.z + position_encoding[sub_nodes]
            z_tgt_pe = self.z + position_encoding[node].repeat(sub_context_embedding.size(0), 1)
            
            context_embedding_aggregate = torch.cat([sub_context_embedding, z_ctx_pe, z_tgt_pe], dim=1)

            batch_pred = self.predictor(context_embedding_aggregate)
            sub_target_embedding = target_embedding[node].unsqueeze(0).repeat(batch_pred.size(0), 1)
            
            batch_loss = criterion(batch_pred, sub_target_embedding)
            loss += batch_loss

        return loss
    
    def forward_legacy(self, data, masked_data, edge_index, target_nodes):
        criterion = nn.MSELoss()
        
        x, masked_x = data.x, masked_data
        pos_enc = data.laplacian_eigenvector_pe
        position_encoding = self.pe_projector(pos_enc)

        with torch.no_grad():
            target_x = self.target_encoder(x, edge_index).detach()

        target_embeddings = target_x[target_nodes]

        total_context_embedding = self.context_encoder(masked_x, edge_index)
        
        total_pred = []
        loss = 0
        for node in target_nodes:
            sub_nodes, sub_edge_index, _,_ = k_hop_subgraph(int(node), num_hops=2, edge_index=edge_index, relabel_nodes=True)
            
            # subgraph_x = masked_x[sub_nodes]
            # subgraph_edge_index = sub_edge_index
            
            # context_embedding = self.context_encoder(subgraph_x, subgraph_edge_index)
            context_embedding = total_context_embedding[sub_nodes]

            z_ctx_pe = self.z + position_encoding[sub_nodes]
            z_tgt_pe = self.z + position_encoding[node].repeat(context_embedding.size(0), 1)
            
            context_embedding_aggregate = torch.cat([context_embedding, z_ctx_pe, z_tgt_pe], dim=1)

            batch_pred = self.predictor(context_embedding_aggregate)
            # print(batch_pred.size(), target_x[node].size())
            target_embedding = target_x[node].unsqueeze(0).repeat(batch_pred.size(0), 1)
            
            batch_loss = criterion(batch_pred, target_embedding)
            # loss += batch_loss / len(batch_pred)
            loss += batch_loss
        
        # context_embeddings = torch.stack(context_embeddings, dim=0)
        # total_pred = torch.stack(total_pred, dim=0)
        # predictor
        return loss
    
    def cosine_similarity(self, x, y):
        return 1 - F.cosine_similarity(x, y, dim=-1).mean()
        
    @torch.no_grad()
    def update_target_encoder(self):
        # print(self.context_encoder.parameters().size(), self.target_encoder.parameters().size())
        for context_params, target_params in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = target_params.data * self.ema + context_params.data * (1 - self.ema)