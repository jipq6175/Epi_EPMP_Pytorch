# paired graph data format and loader

import os
import torch

import numpy as np

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, Linear
from torch_geometric.utils import add_self_loops

from torch.nn import Sequential, Tanh, BatchNorm1d


# data class
class AbAgPair(Data): 
    
    def __init__(self, antigen, antibody): 
        super().__init__()
        
        # antigen: add prefix 'ag' to all fields
        ag_attrs = ['edge_index', 'x']
        if antigen is not None: 
            for ag_attr in ag_attrs: 
                setattr(self, f'ag_{ag_attr}', getattr(antigen, ag_attr))
        
        # antibody: add prefix 'ab' to all fields
        ab_attrs = ['edge_index', 'x']
        if antibody is not None: 
            for ab_attr in ab_attrs: 
                setattr(self, f'ab_{ab_attr}', getattr(antibody, ab_attr))
    
    # incremental operation: for minibatching
    # the index incremental follows # of nodes of antibody or antigen
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ag_edge_index': return self.ag_x.shape[0]
        elif key == 'ab_edge_index': return self.ab_x.shape[0]
        else: return super().__inc__(key, value, *args, **kwargs)



        
def sample_balanced_indices(y): 
    pos_ix = torch.nonzero(y).view(-1).cpu()
    n_pos = pos_ix.shape[0]
    neg_indices = torch.nonzero(y != 1).view(-1)
    neg_ix = torch.tensor(np.random.choice(neg_indices.cpu().numpy(), min(n_pos, neg_indices.shape[0]), replace=False))
    return torch.cat((pos_ix, neg_ix))


def in_batch_sampling(y, batch): 

    ubatch = batch.unique()
    idx, sampled_batch = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
    offset = 0
    for b in ubatch: 
        yb = y[batch == b]

        ix = sample_balanced_indices(yb) + offset

        idx = torch.cat([idx, ix])
        offset += yb.shape[0]
 
    return idx, sampled_batch 


# create bipartite using index and attributes
# with device consistent tensors
def bipartite(ag_x, ag_x_batch, ab_x, ab_x_batch): 
    
    assert ag_x.device == ag_x_batch.device == ab_x.device == ab_x_batch.device
    device = ag_x.device
    
    # node attr and edge index
    x, edge_index = torch.tensor([], dtype=torch.float, device=device), torch.tensor([], dtype=torch.long, device=device)
    # antigen and antibody index for later retreval of the matrix
    ag_index, ab_index = torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.long, device=device)
    
    # getting the unique batch index for the antibody and antigens
    ag_batch_unique, ab_batch_unique = ag_x_batch.unique(), ab_x_batch.unique()
    assert ab_batch_unique.shape[0] == ag_batch_unique.shape[0]
    
    # creating the bipartite following the batch index
    for b in ag_batch_unique: 
        idx_ag, idx_ab = torch.eq(ag_x_batch, b), torch.eq(ab_x_batch, b)
        n_ag, n_ab = idx_ag.sum(), idx_ab.sum()
        
        # keeping the offset
        offset = x.shape[0]
        
        # antigen
        ag = torch.tensor(range(offset, offset + n_ag), dtype=torch.long, device=device)
        ag_index = torch.cat([ag_index, ag])
        # antibody
        ab = torch.tensor(range(offset + n_ag, offset + n_ag + n_ab), dtype=torch.long, device=device)
        ab_index = torch.cat([ab_index, ab])
        
        # construct the edge_index
        edge_index = torch.cat([edge_index, torch.cat([ag.repeat_interleave(n_ab).view(1, -1), ab.repeat(n_ag).view(1, -1)], dim=0)], dim=1)
        edge_index = add_self_loops(edge_index)[0]
        
        # construct the node feature
        x = torch.cat([x, ag_x[idx_ag], ab_x[idx_ab]], dim=0)
        
    return x, edge_index, ag_index, ab_index


class EpiEPMP(torch.nn.Module): 
    
    def __init__(self, node_attr_dim, edge_attr_dim, hidden_dim, h1_dim, h2_dim, share_weight=False, dropout=0.2, heads=4): 
        super(EpiEPMP, self).__init__()
        
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim # dim after the first gnn
        self.h2_dim = h2_dim # dim after the bipartite
        self.share_weight = share_weight
        self.dropout = dropout
        self.heads = heads
        
        self.ag_gnn1 = GCNConv(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn2 = GCNConv(self.hidden_dim, self.h1_dim)
        
        if self.share_weight: 
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else: 
            self.ab_gnn1 = GCNConv(self.node_attr_dim, self.hidden_dim)
            self.ab_gnn2 = GCNConv(self.hidden_dim, self.h1_dim)
        
        self.ag_bnorm1 = BatchNorm1d(self.hidden_dim)
        self.ag_bnorm2 = BatchNorm1d(self.h1_dim)
        self.ab_bnorm1 = BatchNorm1d(self.hidden_dim)
        self.ab_bnorm2 = BatchNorm1d(self.h1_dim)
        
        self.bp_gnn1 = GATConv(self.h1_dim, self.hidden_dim, heads=self.heads, concat=True, dropout=self.dropout)
        self.bp_gnn2 = GATConv(self.hidden_dim * self.heads, self.h2_dim, heads=self.heads, concat=False, dropout=self.dropout)
        
        self.ag_classifier = Sequential(Linear(self.h1_dim + self.h2_dim, self.hidden_dim), 
                                        Tanh(),
                                        Linear(self.hidden_dim, 1))
        
        # Classifier can be shared or not
        # The same as the GCN's
        if self.share_weight: 
            self.ab_classifier = self.ag_classifier
        else: 
            self.ab_classifier = Sequential(Linear(self.h1_dim + self.h2_dim, self.hidden_dim), 
                                            Tanh(),
                                            Linear(self.hidden_dim, 1))
    
    
    def forward(self, ag_x, ag_edge_index, ag_x_batch, \
                      ab_x, ab_edge_index, ab_x_batch):
        
        # antigen gnn + batchnorm
        ag_h1 = self.ag_bnorm1(torch.tanh(self.ag_gnn1(ag_x, ag_edge_index)))
        ag_h1 = self.ag_bnorm2(torch.tanh(self.ag_gnn2(ag_h1, ag_edge_index)))
        
        # antibody gnn + batchnorm
        ab_h1 = self.ab_bnorm1(torch.tanh(self.ab_gnn1(ab_x, ab_edge_index)))
        ab_h1 = self.ab_bnorm2(torch.tanh(self.ab_gnn2(ab_h1, ab_edge_index)))
        
        # bipartite
        x, edge_index, ag_index, ab_index = bipartite(ag_h1, ag_x_batch, ab_h1, ab_x_batch)
        h2 = torch.tanh(self.bp_gnn1(x, edge_index))
        h2 = torch.tanh(self.bp_gnn2(h2, edge_index))
        ag_h2, ab_h2 = h2[ag_index], h2[ab_index]
        
        # skip connection and clasifier
        ag_out = self.ag_classifier(torch.cat([ag_h1, ag_h2], dim=1))
        ab_out = self.ab_classifier(torch.cat([ab_h1, ab_h2], dim=1))
        
        return ag_out, ag_h1, ag_h2, ab_out, ab_h1, ab_h2
        
        
        
