from torch.utils.data import Dataset
import timeit
import os
import lmdb
import numpy as np
import json
import pickle
import dgl
from graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from data_utils import process_files, save_to_file
from graph_sampler import *
import pdb
from path_sampler import *

def add_noise_to_kg(adj_list, triplets, mda_pos_edges, noise_ratio=0.05, seed=42, exclude_rel=39):
    np.random.seed(seed)
    kg_mask = triplets['train'][:, 2] != exclude_rel
    kg_edges = triplets['train'][kg_mask]
    num_noise = int(len(kg_edges) * noise_ratio)
    num_entities = adj_list[0].shape[0]
    num_rels = len(adj_list)
    
    # 可用关系类型（排除 MDA 关系）
    available_rels = [r for r in range(num_rels) if r != exclude_rel]
    if not available_rels:
        print("No available relation types for noise (only MDA relation exists).")
        return adj_list, triplets
    
    existing = set(tuple(e) for e in triplets['train'])
    mda_set = set(tuple(e) for e in mda_pos_edges)
    
    new_edges = []
    new_adj_updates = []
    while len(new_edges) < num_noise:
        head = np.random.randint(0, num_entities)
        tail = np.random.randint(0, num_entities)
        rel = np.random.choice(available_rels)
        if head == tail:
            continue
        if (head, rel, tail) in existing or (head, rel, tail) in mda_set:
            continue
        new_edges.append([head, tail, rel])
        existing.add((head, rel, tail))
        new_adj_updates.append((rel, head, tail))
    
    if not new_edges:
        print("No noise edges added.")
        return adj_list, triplets
    
    # 更新 triplets['train']
    new_edges_np = np.array(new_edges, dtype=int)
    triplets['train'] = np.vstack([triplets['train'], new_edges_np])
    
    # 更新 adj_list：向对应关系类型的邻接矩阵添加边
    for rel, h, t in new_adj_updates:
        # 获取当前关系矩阵的 COO 格式
        coo = adj_list[rel].tocoo()
        rows = np.append(coo.row, h)
        cols = np.append(coo.col, t)
        data = np.append(coo.data, 1)
        # 重建矩阵（注意可能有重复边，但已去重，不会重复）
        adj_list[rel] = scipy.sparse.csc_matrix((data, (rows, cols)), shape=adj_list[rel].shape)
    
    print(f"Added {len(new_edges)} noise edges to KG (ratio={noise_ratio})")
    return adj_list, triplets

def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None, is_ent2rels=None):
    adj_list, triplets, _, _, _, _, _, _ = process_files(params.file_paths,params.node_type1, params.node_type2)
    
    print(f"Before: Total train triplets count = {len(triplets['train'])}")
    #for rel_id in range(len(adj_list)):
    #    print(f"Relation {rel_id}: {adj_list[rel_id].nnz} edges")
    if params.noise_ratio>0:  
        all_mda = np.loadtxt('../data/edges_mda.txt', dtype=int)
        adj_list, triplets = add_noise_to_kg(adj_list, triplets, all_mda, noise_ratio=params.noise_ratio)
    print(f"After: Total train triplets count = {len(triplets['train'])}")
    # 检查特定关系类型的边数变化
    #for rel_id in range(len(adj_list)):
    #    print(f"Relation {rel_id}: {adj_list[rel_id].nnz} edges")
    
    graphs = {}
    for split_name in splits:
        triplets[split_name]=triplets[split_name][triplets[split_name][:,-1]==params.edge_type]
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}

    # Sample train and valid/test links
    for split_name, split in graphs.items():
        print(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, params, split_name,
                                                split['triplets'], params.num_neg_samples_per_link, 
                                                max_size=split['max_size'], constrained_neg_prob=params.constrained_neg_prob)
    
    if not os.path.isdir(params.db_path):
        print('subgraph...........')
        links2subgraphs(adj_list, graphs, params, max_label_value)
    
    if not os.path.isdir(params.db_path_rw):
        print('path...........')
        links2paths(adj_list, graphs, params, max_label_value)
        
class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, params,included_relations=None, 
                 num_neg_samples_per_link=1, dataset='', kge_model='', file_name='', is_ret_nodes_num=False):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features = None
        self.kge_entity2id = None
        
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name
        self.is_ret_nodes_num = is_ret_nodes_num

        ssp_graph, __, __, __, id2entity, id2relation, m_h2r, m_t2r = process_files(raw_data_paths, 
                                                                                        params.node_type1,params.node_type2)
        self.num_rels = len(ssp_graph)

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.m_h2r = m_h2r
        self.m_t2r = m_t2r

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        print('Max distance from sub : ',self.max_n_label[0],' Max distance from obj : ',self.max_n_label[1])

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            #print(deserialize(txn.get(str_id)).values())
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            #print(nodes_pos, r_label_pos, g_label_pos, n_labels_pos)
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
            
            # Get the neighbor relations of target head and tails
            # nei_rels_pos = [self.ent2rels[nodes_pos[0]], self.ent2rels[nodes_pos[1]]]
            nei_rels_pos = [[0, 1], [0, 1]]
            
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        nei_rels_negs = []

        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                # Get the neighbor relations of target head and tails
                # nei_rels_neg = [self.ent2rels[nodes_neg[0]], self.ent2rels[nodes_neg[1]]]
                nei_rels_neg = [[0, 1], [0, 1]]
                nei_rels_negs.append(nei_rels_neg)
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)
        
        #print("Nodes of subgraph: ", len(subgraph_pos.nodes()))
        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg, 

    def __len__(self):
        return self.num_graphs_pos
    
    def _prepare_subgraphs(self,nodes, r_label, n_labels):
        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        #print(subgraph.edata['type'])
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        
        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)
        
        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, r_label, n_feats)

        # Add the original node id feature
        subgraph.ndata['parent_id'] = self.graph.subgraph(nodes).parent_nid
        # Add the neighbor relations 
        subgraph.ndata['out_nei_rels'] = torch.LongTensor(self.m_h2r[subgraph.ndata['parent_id']])
        subgraph.ndata['in_nei_rels'] = torch.LongTensor(self.m_t2r[subgraph.ndata['parent_id']])
        
        #subgraph.ndata['node_pair'] = torch.LongTensor(str_id)
        #print(subgraph.ndata)
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, r_label, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        
        # 'id' is used to represent the target head and targe tail nodes
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)
        
        # 'r_label' is used to represent the relation label of this subgraph
        subgraph.ndata['r_label'] = torch.LongTensor(np.ones(n_nodes) * r_label)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim

        
        return subgraph

class PathDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, params,included_relations=None, 
                 num_neg_samples_per_link=1, dataset='', kge_model='', file_name='', is_ret_nodes_num=False):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features = None
        self.kge_entity2id = None
        
        # self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name
        self.is_ret_nodes_num = is_ret_nodes_num

        ssp_graph, __, __, __, id2entity, id2relation, m_h2r, m_t2r = process_files(raw_data_paths, 
                                                                                        params.node_type1,params.node_type2)
        self.num_rels = len(ssp_graph)

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.m_h2r = m_h2r
        self.m_t2r = m_t2r

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        print('Max distance from sub : ',self.max_n_label[0],' Max distance from obj : ',self.max_n_label[1])

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            #print(index)
            str_id = '{:08}'.format(index).encode('ascii')
            #print(str_id)
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            #print(nodes_pos, r_label_pos, g_label_pos, n_labels_pos)
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
            
            # Get the neighbor relations of target head and tails
            # nei_rels_pos = [self.ent2rels[nodes_pos[0]], self.ent2rels[nodes_pos[1]]]
            nei_rels_pos = [[0, 1], [0, 1]]
            
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        nei_rels_negs = []

        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                # Get the neighbor relations of target head and tails
                # nei_rels_neg = [self.ent2rels[nodes_neg[0]], self.ent2rels[nodes_neg[1]]]
                nei_rels_neg = [[0, 1], [0, 1]]
                nei_rels_negs.append(nei_rels_neg)
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)
        
        #print("Nodes of subgraph: ", len(subgraph_pos.nodes()))
        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg, 

    def __len__(self):
        return self.num_graphs_pos
    
    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        #print(subgraph.edata['type'])
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        
        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)
        
        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, r_label, n_feats)

        # Add the original node id feature
        subgraph.ndata['parent_id'] = self.graph.subgraph(nodes).parent_nid
        # Add the neighbor relations 
        subgraph.ndata['out_nei_rels'] = torch.LongTensor(self.m_h2r[subgraph.ndata['parent_id']])
        subgraph.ndata['in_nei_rels'] = torch.LongTensor(self.m_t2r[subgraph.ndata['parent_id']])

        #subgraph.ndata['node_pair'] = torch.LongTensor(nodes)
        return subgraph
    def _prepare_features_new(self, subgraph, n_labels, r_label, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        
        # 'id' is used to represent the target head and targe tail nodes
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)
        
        # 'r_label' is used to represent the relation label of this subgraph
        subgraph.ndata['r_label'] = torch.LongTensor(np.ones(n_nodes) * r_label)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim

        
        return subgraph