import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from scipy.sparse import SparseEfficiencyWarning
from datasets import SubgraphDataset, generate_subgraph_datasets, PathDataset
from model.graph_classifier import GraphClassifier as dgl_model
from evaluator import Evaluator
from trainer import Trainer
from warnings import simplefilter
from parameters import args_parser
import torch
import json
import numpy as np
import pickle
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=SparseEfficiencyWarning)

def build_index_map(data):
    index_map={}
    for idx, (head, edge_type, tail) in enumerate(data):
        if edge_type not in index_map:
            index_map[edge_type] = []
        index_map[edge_type].append(head)
        index_map[edge_type].append(tail)
    for edge_type in index_map.keys():
        index_map[edge_type]=list(set(index_map[edge_type]))
    return index_map
    
if __name__ == '__main__':
    params = args_parser()
    print(params.dataset)
    index_map={}
    data1=[]
    data=np.loadtxt('../data/'+params.dataset+'/train.txt',delimiter='\t').astype(int)
    data2=np.loadtxt('../data/'+params.dataset+'/valid.txt',delimiter='\t').astype(int)
    if not os.path.exists('index_map_'+params.dataset+'.pkl'):
        index_map = build_index_map(np.vstack((data,data2)))
        with open('index_map_'+params.dataset+'.pkl', 'wb') as f:
            pickle.dump(index_map, f)
    
    params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.negative}_hop_{params.hop}')
    params.db_path_rw = os.path.join(params.main_dir, f'data/{params.dataset}/path_neg_{params.negative}')
    
    if not (os.path.isdir(params.db_path) and os.path.isdir(params.db_path_rw)):
        generate_subgraph_datasets(params)
    
    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,params,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file)
    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths,params,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file)

    train_path = PathDataset(params.db_path_rw, 'train_pos', 'train_neg', params.file_paths,params,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file)
    valid_path = PathDataset(params.db_path_rw, 'valid_pos', 'valid_neg', params.file_paths,params,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file)
    
    # Avoid that m_h2r and m_t2r of train and valid are different
    valid.m_h2r = train.m_h2r
    valid.m_t2r = train.m_t2r
    
    valid_path.m_h2r = train_path.m_h2r
    valid_path.m_t2r = train_path.m_t2r

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.train_path_dim = train_path.n_feat_dim
    params.train_dim = train.n_feat_dim
    #print(train.n_feat_dim,train_path.n_feat_dim,valid.n_feat_dim,valid_path.n_feat_dim)
    # Set the embedding dimension of relation and node
    if params.init_nei_rels == 'no':
        params.inp_dim = train.n_feat_dim
    else:
        params.inp_dim = train.n_feat_dim + params.sem_dim

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label
    
    
    #params.graph_feat = 32
    #node_feat=torch.FloatTensor(np.random.uniform(0,1,(66481,params.graph_feat))).to(device=params.device)
    #data=torch.LongTensor(data.T).to(device=params.device)
    #data2=torch.LongTensor(data2.T).to(device=params.device)
    
    #pair_train_feat=torch.FloatTensor(np.random.uniform(0,1,(len(train),params.graph_feat))).to(device=params.device)
    #pair_valid_feat=torch.FloatTensor(np.random.uniform(0,1,(len(valid),params.graph_feat))).to(device=params.device)
    
    graph_classifier = dgl_model(params).to(device=params.device)

    print(f"Device: {params.device}")
    print(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")
    valid_evaluator = Evaluator(params, graph_classifier, valid, valid_path)
    trainer = Trainer(params, graph_classifier, train, train_path, valid_evaluator)
    print('Starting training with full batch...')
    trainer.train()
    
    
