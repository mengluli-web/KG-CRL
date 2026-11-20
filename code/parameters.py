# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:29:53 2024

@author: lml
"""

import argparse
import os
import logging
import json
import torch
import random
import numpy as np
from graph_utils import collate_dgl_train, move_batch_to_device_dgl_train

def initialize_experiment(params, file_name):
    '''
    Makes the experiment directory, sets standard paths and initializes the logger
    '''
    params.main_dir = os.path.join('../')
    exps_dir = os.path.join('experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    params.exp_dir = os.path.join(exps_dir, params.experiment_name)

    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)

    if file_name == 'test.py':
        params.test_exp_dir = os.path.join(params.exp_dir, f"test_{params.dataset}_{params.constrained_neg_prob}")
        if not os.path.exists(params.test_exp_dir):
            os.makedirs(params.test_exp_dir)
        file_handler = logging.FileHandler(os.path.join(params.test_exp_dir, f"log_test.txt"))
    else:
        file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log_train.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)


def args_parser():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument('--seeds', type=int, default=42)
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,default='MDA_indep',
                        help="Dataset string")
    parser.add_argument("--negative", type=str, default='random_all') ## random_microbe, random_disease
    parser.add_argument("--output", type=str,default='result/result')
                        
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")

    # Training regime params
    parser.add_argument("--noise_ratio", type=float, default=0.0)
    parser.add_argument("--num_epochs", "-ne", type=int, default=20)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=5e-4,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="loss parameter of combine sub and path")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="loss parameter of sub")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="loss parameter of path")
    parser.add_argument("--omega", type=float, default=1.0,
                        help="loss parameter of contrastive learning")
    
    parser.add_argument("--kg", type=str, default='with_kg')
    parser.add_argument("--crl", type=str, default='with_crl')
    parser.add_argument("--sub", type=str, default='with_subgraph')
    parser.add_argument("--path", type=str, default='with_path')
    parser.add_argument("--CLoss", type=str, default='with_CLoss')
                               
    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=1,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=100,
                        help="if > 0, upper bound the # nodes per hop by subsampling") #None
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=False,
                        help='whether to only consider enclosing subgraph')#True
      
    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32,
                        help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=1,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.0,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True,
                        help='whether to have attn in model or not')
    parser.add_argument('--sem_dim', type=int, default=24,
                        help='the dimension of sematic part of node embedding')
    parser.add_argument('--max_nei_rels', type=int, default=10, help='the maximum num of neighbor relations of each node when initialzing the node embedding.')#10
    parser.add_argument('--nei_rels_dropout', type=float, default=0.4, help='Dropout rate in aggregating relation embeddings.')#0.4
    parser.add_argument('--is_comp', type=str, default='mult', choices=['mult', 'sub'], help='The composition manner of node and relation')
    parser.add_argument('--comp_ht', type=str, choices=['mult, mlp, sum'], default='sum', help='The composition operator of head and tail embedding')
    parser.add_argument('--comp_hrt', type=str, choices=['TransE, DistMult'], default='TransE', help='The composition operator of (h, r, t)embedding')
    parser.add_argument('--init_nei_rels', type=str, choices=['no', 'out', 'in', 'both'], default='in', help='the manner of utilizing relatioins when initializing entity embedding')
    parser.add_argument('--nei_rel_path', type=bool, default=True,
                        help='whether to consider neighboring relational paths')
    parser.add_argument('--path_agg', type=str, choices=['mean', 'att'], default='att', help='the manner of aggreating neighboring relational paths.')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--cat_or_add', type=str, default="add")#cat
    parser.add_argument('--with_random', type=str2bool, default=True)
    parser.add_argument('--eval_random', type=str2bool, default=False)
    
    parser.add_argument('--edge_type', type=int, default=39)
    
    params = parser.parse_args()
    initialize_experiment(params, __file__)
    
    if params.kg=='without_kg':
        params.dataset=params.dataset+'_reindex2'
        params.edge_type = 0
    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_file))
    }
    
    seed=params.seeds
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    else:
        params.device = torch.device('cpu')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    if 'MDA' in params.dataset:
        params.task = 'mda'
        params.node_type1 = 'microbe'
        params.node_type2 = 'disease'
        #params.edge_type = 39
    params.collate_fn = collate_dgl_train
    params.move_batch_to_device = move_batch_to_device_dgl_train
    return params