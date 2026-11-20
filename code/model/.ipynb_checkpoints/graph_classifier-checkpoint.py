from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from functools import partial
import torch.nn.functional as F
import torch
import time
import numpy as np
from .discriminator import Discriminator
from .batch_gru import BatchGRU
import dgl
import scipy.sparse as sp
import copy
from copy import deepcopy
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor

class GraphClassifier(nn.Module):
    def __init__(self, params):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()
        
        self.params = params
        hidden=self.params.num_gcn_layers * self.params.emb_dim
        hidden_out = self.params.num_classes
        
        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        
        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.params.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.params.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)
                
        self.out_dim = self.params.num_gcn_layers * self.params.emb_dim
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.fc_path = nn.Linear(params.train_path_dim, params.train_dim)
        # num_rels + 1 instead of nums_rels, in order to add a "padding" relation.
        self.rel_emb = nn.Embedding(self.params.num_rels + 1, self.params.inp_dim, sparse=False, padding_idx=self.params.num_rels)
        
        self.ent_padding = nn.Parameter(torch.FloatTensor(1, self.params.sem_dim).uniform_(-1, 1))
        if self.params.init_nei_rels == 'both':
            self.w_rel2ent = nn.Linear(2 * self.params.inp_dim, self.params.sem_dim)
        elif self.params.init_nei_rels == 'out' or 'in':
            self.w_rel2ent = nn.Linear(self.params.inp_dim, self.params.sem_dim)

        self.sigmoid = nn.Sigmoid()
        self.nei_rels_dropout = nn.Dropout(self.params.nei_rels_dropout)
        self.dropout = nn.Dropout(self.params.dropout)
        self.softmax = nn.Softmax(dim=1)
        
        self.fc_nodes = nn.Linear(params.train_dim, self.params.num_gcn_layers * self.params.emb_dim)
        
        self.fc_comp = nn.Linear(2 * self.params.emb_dim, self.params.emb_dim)
        
        #self.fc_layer = nn.Linear(2 * self.params.num_gcn_layers * self.params.emb_dim, hidden_out)
        self.fc_layer = nn.Linear(2 * self.params.num_gcn_layers * self.params.emb_dim, self.params.emb_dim)
        self.fc_layer2 = nn.Linear(self.params.emb_dim, hidden_out)
        
        self.fc_layer_a = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim, hidden_out)
            
        if self.params.comp_ht == 'mlp':
            self.fc_comp = nn.Linear(2 * self.params.emb_dim, self.params.emb_dim)

        self.rnn = torch.nn.GRU(self.params.emb_dim, self.params.emb_dim, batch_first=True)

        self.batch_gru = BatchGRU(self.params.num_gcn_layers * self.params.emb_dim )

        self.W_o = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim*2, self.params.num_gcn_layers * self.params.emb_dim)
        #self.W_o = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim, self.params.num_gcn_layers * self.params.emb_dim)
        
    def init_ent_emb_matrix(self, g):
        """ Initialize feature of entities by matrix form """
        out_nei_rels = g.ndata['out_nei_rels']
        in_nei_rels = g.ndata['in_nei_rels']
        
        target_rels = g.ndata['r_label']
        out_nei_rels_emb = self.rel_emb(out_nei_rels)
        in_nei_rels_emb = self.rel_emb(in_nei_rels)
        target_rels_emb = self.rel_emb(target_rels).unsqueeze(2)

        out_atts = self.softmax(self.nei_rels_dropout(torch.matmul(out_nei_rels_emb, target_rels_emb).squeeze(2)))
        in_atts = self.softmax(self.nei_rels_dropout(torch.matmul(in_nei_rels_emb, target_rels_emb).squeeze(2)))
        out_sem_feats = torch.matmul(out_atts.unsqueeze(1), out_nei_rels_emb).squeeze(1)
        in_sem_feats = torch.matmul(in_atts.unsqueeze(1), in_nei_rels_emb).squeeze(1)
        
        if self.params.init_nei_rels == 'both':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(torch.cat([out_sem_feats, in_sem_feats], dim=1)))
        elif self.params.init_nei_rels == 'out':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(out_sem_feats))
        elif self.params.init_nei_rels == 'in':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(in_sem_feats))
        #if ent_sem_feats.size(0)==self.params.train_path_dim:
        #    ent_sem_feats=F.relu(self.fc_path2(ent_sem_feats))
            
        g.ndata['init'] = torch.cat([g.ndata['feat'], ent_sem_feats], dim=1)  # [B, self.inp_dim]

    def comp_ht_emb(self, head_embs, tail_embs):
        if self.params.comp_ht == 'mult':
            ht_embs = head_embs * tail_embs
        elif self.params.comp_ht == 'mlp':
            ht_embs = self.fc_comp(torch.cat([head_embs, tail_embs], dim=1))
        elif self.params.comp_ht == 'sum':
            ht_embs = head_embs + tail_embs
        else:
            raise KeyError(f'composition operator of head and relation embedding {self.comp_ht} not recognized.')

        return ht_embs

    def comp_hrt_emb(self, head_embs, tail_embs, rel_embs):
        rel_embs = rel_embs.repeat(1, self.params.num_gcn_layers)
        if self.params.comp_hrt == 'TransE':
            hrt_embs = head_embs + rel_embs - tail_embs
        elif self.params.comp_hrt == 'DistMult':
            hrt_embs = head_embs * rel_embs * tail_embs
        else: raise KeyError(f'composition operator of (h, r, t) embedding {self.comp_hrt} not recognized.')
        
        return hrt_embs

    def nei_rel_path(self, g, rel_labels, r_emb_out):
        """ Neighboring relational path module """
        # Only consider in-degree relations first.
        nei_rels = g.ndata['in_nei_rels']
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        heads_rels = nei_rels[head_ids]
        tails_rels = nei_rels[tail_ids]

        # Extract neighboring relational paths
        batch_paths = []
        for (head_rels, r_t, tail_rels) in zip(heads_rels, rel_labels, tails_rels):
            paths = []
            for h_r in head_rels:
                for t_r in tail_rels:
                    path = [h_r, r_t, t_r]
                    paths.append(path)
            batch_paths.append(paths)       # [B, n_paths, 3] , n_paths = n_head_rels * n_tail_rels
        
        batch_paths = torch.LongTensor(batch_paths).to(rel_labels.device)# [B, n_paths, 3], n_paths = n_head_rels * n_tail_rels
        batch_size = batch_paths.shape[0]
        batch_paths = batch_paths.view(batch_size * len(paths), -1) # [B * n_paths, 3]

        batch_paths_embs = F.embedding(batch_paths, r_emb_out, padding_idx=-1) # [B * n_paths, 3, inp_dim]

        # Input RNN 
        _, last_state = self.rnn(batch_paths_embs) # last_state: [1, B * n_paths, inp_dim]
        last_state = last_state.squeeze(0) # squeeze the dim 0 
        last_state = last_state.view(batch_size, len(paths), self.params.emb_dim) # [B, n_paths, inp_dim]
        # Aggregate paths by attention
        if self.params.path_agg == 'mean':
            output = torch.mean(last_state, 1) # [B, inp_dim]
        
        if self.params.path_agg == 'att':
            r_label_embs = F.embedding(rel_labels, r_emb_out, padding_idx=-1) .unsqueeze(2) # [B, inp_dim, 1]
            atts = torch.matmul(last_state, r_label_embs).squeeze(2) # [B, n_paths]
            atts = F.softmax(atts, dim=1).unsqueeze(1) # [B, 1, n_paths]
            output = torch.matmul(atts, last_state).squeeze(1) # [B, 1, n_paths] * [B, n_paths, inp_dim] -> [B, 1, inp_dim] -> [B, inp_dim]
        else:
            raise ValueError('unknown path_agg')
        
        return output # [B, inp_dim]

    def get_logits(self, s_G, s_g_pos, s_g_cor): 
        ret = self.disc(s_G, s_g_pos, s_g_cor)
        return ret
    
    def split_ca_co(self,g,threshold=0.5):
        graphs_pos_c=[]
        graphs_pos_o=[]
        unbatched_graphs = dgl.unbatch(g)
        #print(unbatched_graphs)
        for graph in unbatched_graphs:
            #print(graph.edges())
            edges_below_threshold = graph.edata['weight'] < threshold
            edges_above_threshold = graph.edata['weight'] >= threshold
            
            below_indices = edges_below_threshold.nonzero(as_tuple=True)[0]
            above_indices = edges_above_threshold.nonzero(as_tuple=True)[0]
            
            if above_indices.is_cuda:
                above_indices = above_indices.cpu()
            if below_indices.is_cuda:
                below_indices = below_indices.cpu()
                
            src_below, dst_below = graph.edges(form='uv', order='eid')[0][below_indices], graph.edges(form='uv', order='eid')[1][below_indices]
            src_above, dst_above = graph.edges(form='uv', order='eid')[0][above_indices], graph.edges(form='uv', order='eid')[1][above_indices]
            
            subgraph_below = dgl.DGLGraph()
            subgraph_below.add_nodes(len(graph.nodes()))
            subgraph_below.ndata.update({key: graph.ndata[key] for key in graph.ndata.keys()})
            subgraph_below.add_edges(src_below, dst_below)
            subgraph_below.edata.update({key: graph.edata[key][below_indices] for key in graph.edata.keys()})
            graphs_pos_c.append(subgraph_below)
            
            subgraph_above = dgl.DGLGraph()
            subgraph_above.add_nodes(len(graph.nodes()))
            subgraph_above.ndata.update({key: graph.ndata[key] for key in graph.ndata.keys()})
            subgraph_above.add_edges(src_above, dst_above)
            subgraph_above.edata.update({key: graph.edata[key][above_indices] for key in graph.edata.keys()})
            graphs_pos_o.append(subgraph_above)
        return dgl.batch(graphs_pos_c),dgl.batch(graphs_pos_o)
    
    def GNN_encoder(self, graph,graph_sizes,reprs):
        graph.ndata['repr']=reprs
        r = self.rel_emb.weight.clone()
        graph.ndata['h'], r_emb_out = self.gnn(graph, r)
        
        graph.ndata['repr'] = F.relu(self.batch_gru(graph.ndata['repr'].view(-1, self.out_dim), graph_sizes))
        node_hiddens = F.relu(self.W_o(graph.ndata['repr']))
        #node_hiddens = F.relu(self.W_o(graph.ndata['repr'].view(-1, self.out_dim)))  # num_nodes x hidden 
        graph.ndata['repr'] = self.dropout(node_hiddens)
        return graph,r_emb_out
        
    def get_embeds(self, data, graph_sizes,eval_random, cor_graph=False,pathes=False):
        g, rel_labels = data
        #print(g)
        if g.ndata['feat'].size(1)==self.params.train_path_dim:
            g.ndata['feat']=F.relu(self.fc_path(g.ndata['feat']))
        
        if self.params.init_nei_rels == 'no':
            g.ndata['init'] = g.ndata['feat'].clone()
        else:
            self.init_ent_emb_matrix(g)
        '''
        if cor_graph:
            g.ndata['init'] = g.ndata['init'][torch.randperm(g.ndata['feat'].shape[0])]  
        '''
        
        g,r_emb_out=self.GNN_encoder(g,graph_sizes,g.ndata['feat'])
        #g.ndata['repr']=F.relu(self.fc_nodes(g.ndata['feat']))
        
        row, col = g.edges(form='all')[0],g.edges(form='all')[1]
        g.edata['repr'] = torch.cat([g.ndata['repr'][row], g.ndata['repr'][col]], dim=-1)
        
        x=g.ndata['repr']
        
        if self.params.crl=='with_crl':
            edge_rep=g.edata['repr']
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
            #print(x.size(),edge_rep.size(),edge_att,edge_att.size())
            edge_weight_c = edge_att[:, 0]
            edge_weight_o = edge_att[:, 1]
            
            #node_att = F.softmax(self.node_att_mlp(x), dim=-1)
            #xc = node_att[:, 0].view(-1, 1) * x
            #xo = node_att[:, 1].view(-1, 1) * x
            
            g.edata['weight']=edge_weight_o
            g_c,g_o=self.split_ca_co(g)
            
            g_c,_=self.GNN_encoder(g_c,graph_sizes,x)
            g_o,_=self.GNN_encoder(g_o,graph_sizes,x)
            #g_c,_=self.GNN_encoder(g_c,graph_sizes,self.bnc(xc))
            #g_o,_=self.GNN_encoder(g_o,graph_sizes,self.bno(xo))
            
            xc = mean_nodes(g_c, 'repr').view(-1, self.out_dim)
            xo = mean_nodes(g_o, 'repr').view(-1, self.out_dim)
            
            return xc, xo
        else:
            x = mean_nodes(g, 'repr').view(-1, self.out_dim)
            return [],x_nc
            
        
    def context_readout_layer(self, x):
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = x#F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):
        num = xc.shape[0]
        l = [i for i in range(num)]
        if self.params.with_random:
            if eval_random:
                np.random.shuffle(l)
        random_idx = torch.tensor(l)
        #print(random_idx)
        if self.params.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = x#F.log_softmax(x, dim=-1)
        return x_logis
        
    def forward(self, data, data_graph_sizes, data_path, path_graph_sizes,data_neg, data_graph_sizes_neg, data_path_neg, path_graph_sizes_neg,eval_random=True, cor_graph=False,trains=True):
        if self.params.crl=='with_crl' and self.params.sub=='with_subgraph' and self.params.path=='with_path':
            ###subgraph-based embeddings
            embed_c_sub_pos,embed_o_sub_pos=self.get_embeds(data, data_graph_sizes,eval_random, cor_graph)
            embed_c_path_pos,embed_o_path_pos=self.get_embeds(data_path, path_graph_sizes,eval_random, cor_graph)
            embed_c_sub_neg,embed_o_sub_neg=self.get_embeds(data_neg, data_graph_sizes_neg,eval_random, cor_graph)
            embed_c_path_neg,embed_o_path_neg=self.get_embeds(data_path_neg, path_graph_sizes_neg,eval_random, cor_graph)
            
            embed_o_sub=torch.cat((embed_o_sub_pos,embed_o_sub_neg),dim=0)
            embed_o_path=torch.cat((embed_o_path_pos,embed_o_path_neg),dim=0)
            #output = self.fc_layer(torch.cat((embed_o_sub,embed_o_path),dim=1))
            ee = F.relu(self.fc_layer(torch.cat((embed_o_sub,embed_o_path),dim=1)))
            output = self.fc_layer2(ee)
            
            embed_c_sub=torch.cat((embed_c_sub_pos,embed_c_sub_neg),dim=0)
            embed_c_path=torch.cat((embed_c_path_pos,embed_c_path_neg),dim=0)
            
            pred_sub_c = self.context_readout_layer(embed_c_sub)
            pred_sub_o = self.objects_readout_layer(embed_o_sub)
            pred_sub_co = self.random_readout_layer(embed_c_sub, embed_o_sub, eval_random=eval_random)
            
            pred_path_c = self.context_readout_layer(embed_c_path)
            pred_path_o = self.objects_readout_layer(embed_o_path)
            pred_path_co = self.random_readout_layer(embed_c_path, embed_o_path, eval_random=eval_random)
            
            pred_subo_pathc = self.random_readout_layer(embed_c_path, embed_o_sub, eval_random=eval_random)
            pred_subc_patho = self.random_readout_layer(embed_c_sub, embed_o_path, eval_random=eval_random)

            if trains:
                return output,[pred_sub_c,pred_sub_o,pred_sub_co],\
                        [pred_path_c,pred_path_o,pred_path_co],\
                        [pred_subo_pathc,pred_subc_patho],\
                        [embed_o_sub_pos,embed_o_path_pos,embed_o_sub_neg,embed_o_path_neg]
            else:
                return F.sigmoid(output)
        elif self.params.crl=='without_crl':
            ###subgraph-based embeddings
            embed_c_sub_pos,embed_o_sub_pos=self.get_embeds(data, data_graph_sizes,eval_random, cor_graph)
            embed_c_path_pos,embed_o_path_pos=self.get_embeds(data_path, path_graph_sizes,eval_random, cor_graph)
            embed_c_sub_neg,embed_o_sub_neg=self.get_embeds(data_neg, data_graph_sizes_neg,eval_random, cor_graph)
            embed_c_path_neg,embed_o_path_neg=self.get_embeds(data_path_neg, path_graph_sizes_neg,eval_random, cor_graph)
            
            embed_o_sub=torch.cat((embed_o_sub_pos,embed_o_sub_neg),dim=0)
            embed_o_path=torch.cat((embed_o_path_pos,embed_o_path_neg),dim=0)
            output = self.fc_layer(torch.cat((embed_o_sub,embed_o_path),dim=1))

            pred_sub_o = self.objects_readout_layer(embed_o_sub)
            pred_path_o = self.objects_readout_layer(embed_o_path)
            pred_sub_c,pred_sub_co,pred_path_c,pred_path_co,pred_subo_pathc,pred_subc_patho=[],[],[],[],[],[]
              
            if trains:
                return output,[pred_sub_c,pred_sub_o,pred_sub_co],\
                        [pred_path_c,pred_path_o,pred_path_co],\
                        [pred_subo_pathc,pred_subc_patho],\
                        [embed_o_sub_pos,embed_o_path_pos,embed_o_sub_neg,embed_o_path_neg]
            else:
                return F.sigmoid(output)
        elif self.params.sub=='without_subgraph':
            ###subgraph-based embeddings
            embed_c_path_pos,embed_o_path_pos=self.get_embeds(data_path, path_graph_sizes,eval_random, cor_graph)
            embed_c_path_neg,embed_o_path_neg=self.get_embeds(data_path_neg, path_graph_sizes_neg,eval_random, cor_graph)
            
            embed_o_path=torch.cat((embed_o_path_pos,embed_o_path_neg),dim=0)
            embed_c_path=torch.cat((embed_c_path_pos,embed_c_path_neg),dim=0)
            pred_path_c = self.context_readout_layer(embed_c_path)
            pred_path_o = self.objects_readout_layer(embed_o_path)
            pred_path_co = self.random_readout_layer(embed_c_path, embed_o_path, eval_random=eval_random)
            if trains:
                return [pred_path_c,pred_path_o,pred_path_co],\
                        [embed_o_path_pos,embed_o_path_neg]
            else:
                return F.sigmoid(pred_path_o)
        elif self.params.path=='without_path':
            ###subgraph-based embeddings
            embed_c_sub_pos,embed_o_sub_pos=self.get_embeds(data, data_graph_sizes,eval_random, cor_graph)
            embed_c_sub_neg,embed_o_sub_neg=self.get_embeds(data_neg, data_graph_sizes_neg,eval_random, cor_graph)
            
            embed_o_sub=torch.cat((embed_o_sub_pos,embed_o_sub_neg),dim=0)
            
            embed_c_sub=torch.cat((embed_c_sub_pos,embed_c_sub_neg),dim=0)
            pred_sub_c = self.context_readout_layer(embed_c_sub)
            pred_sub_o = self.objects_readout_layer(embed_o_sub)
            pred_sub_co = self.random_readout_layer(embed_c_sub, embed_o_sub, eval_random=eval_random)
            if trains:
                return [pred_sub_c,pred_sub_o,pred_sub_co],\
                        [embed_o_sub_pos,embed_o_sub_neg]
            else:
                return F.sigmoid(pred_sub_o)
        
