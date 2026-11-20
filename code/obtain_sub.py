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

from torch.utils.data import Dataset
import timeit
import lmdb
import json
import pickle
import dgl
from graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from data_utils import process_files, save_to_file
from graph_sampler import *
import pdb
from path_sampler import *

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
    
    splits=['train', 'valid']
    saved_relation2id=None
    max_label_value=None
    is_ent2rels=None
    
    adj_list, triplets, _, _, _, _, _, _ = process_files(params.file_paths,params.node_type1, params.node_type2)

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
    
    
    


    