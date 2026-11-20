import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import logging
from scipy.sparse import SparseEfficiencyWarning
from datasets import SubgraphDataset, generate_subgraph_datasets, PathDataset
from model.graph_classifier_indep import GraphClassifier as dgl_model
from evaluator_indep import Evaluator
from trainer_indep import Trainer
from warnings import simplefilter
from parameters import args_parser
import torch
import json
import numpy as np
import pickle

simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=SparseEfficiencyWarning)

def initialize_model(params, model, ent2rels=None, load_model=False):
    if load_model and os.path.exists(os.path.join(params.exp_dir, 'best_graph_classifier.pth')):
        logging.info('Loading existing model from %s' % os.path.join(params.exp_dir, 'best_graph_classifier.pth'))
        graph_classifier = torch.load(os.path.join(params.exp_dir, 'best_graph_classifier.pth')).to(device=params.device)
    else:
        logging.info('No existing model found. Initializing new model..')
        graph_classifier = model(params).to(device=params.device)

    return graph_classifier

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
    #logging.basicConfig(level=logging.INFO)
    params = args_parser()
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
    #print(os.path.isdir(params.db_path),os.path.isdir(params.db_path_rw))

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
    #print(train.n_feat_dim,train_path.n_feat_dim,valid_path.n_feat_dim)
    # Set the embedding dimension of relation and node
    if params.init_nei_rels == 'no':
        params.inp_dim = train.n_feat_dim
    else:
        params.inp_dim = train.n_feat_dim + params.sem_dim

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label

    graph_classifier = initialize_model(params, dgl_model, params.load_model)

    print(f"Device: {params.device}")
    print(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    valid_evaluator = Evaluator(params, graph_classifier, valid, valid_path)

    trainer = Trainer(params, graph_classifier, train, train_path, valid_evaluator)

    print('Starting training with full batch...')

    trainer.train()
    
