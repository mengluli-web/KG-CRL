import os
import math
import struct
import random
import pickle as pkl
import pdb
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import sys
import torch
import scipy  
from scipy.special import softmax
from dgl_utils import _bfs_relational
from graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius
import networkx as nx
from scipy.sparse.csgraph import depth_first_order

def links2paths(A, graphs, params, max_label_value=None):
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    BYTES_PER_DATUM = get_average_path_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5
    links_length = 0
    for split_name, split in graphs.items():
        links_length += (len(split['pos']) + len(split['neg'])) * 2
    map_size = links_length * BYTES_PER_DATUM

    env = lmdb.open(params.db_path_rw, map_size=map_size, max_dbs=6)

    def extraction_helper(A, links, g_labels, split_env):

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (str_id, datum) in tqdm(p.imap(extract_save_path, args_), total=len(links)):
                max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                subgraph_sizes.append(datum['subgraph_size'])
                enc_ratios.append(datum['enc_ratio'])
                num_pruned_nodes.append(datum['num_pruned_nodes'])

                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in graphs.items():
        print(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env)

        print(f"Extracting enclosing subgraphs for negative links in {split_name} set")
        labels = np.zeros(len(split['neg']))
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, split_env)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))


def get_average_path_size(sample_size, links, A, params):
    total_size = 0
    nums=0
    for (n1, n2, r_label) in links:#[np.random.choice(len(links), sample_size)]:
        print(nums,n1,n2)
        nums+=1
        srcs,nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = path_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}#'node_pairs':srcs, 
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(A, params, max_label_value):
    global A_, params_, max_label_value_
    A_, params_, max_label_value_ = A, params, max_label_value


def extract_save_path(args_):
    idx, (n1, n2, r_label), g_label = args_
    srcs,nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = path_extraction_labeling((n1, n2), r_label, A_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop)

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}#'node_pairs':srcs, 
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)

def dfs_path(graph_matrix, start_node, end_node):
    #distances, predecessors = breadth_first_order(graph_matrix, start_node, return_predecessors=True)
    distances, predecessors = depth_first_order(graph_matrix, start_node, return_predecessors=True)
    #print(distances, predecessors, end_node in distances, end_node in predecessors)
    if end_node not in predecessors:  
        return [start_node,end_node], []
    path_nodes = []
    path_edges = []
    current_node = end_node
    while current_node != start_node:
        previous_node = predecessors[current_node]
        path_nodes.append(current_node)
        path_edges.append((previous_node, current_node))
        current_node = previous_node
    if (start_node, end_node) in path_edges:
        path_edges.remove((start_node, end_node))
    #path_nodes.reverse()
    #print([start_node]+path_nodes)
    if end_node in path_nodes:
        path_nodes.remove(end_node)
    path_nodes=[start_node,end_node]+path_nodes
    #print(path_nodes)
    path_edges.reverse()
    return path_nodes, path_edges

def path_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_list[-1][ind[0], ind[1]] = 0
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T 
    subgraph_nodes,subgraph_edges = dfs_path(A_incidence, ind[0], ind[1])

    # np.random.shuffle(subgraph_nodes)
    subgraph = [np.zeros((len(subgraph_nodes), len(subgraph_nodes))) for _ in A_list]
    node_index_dict = {node: index for index, node in enumerate(subgraph_nodes)}
    rows = np.array([node_index_dict[u] for u, v in subgraph_edges])  
    cols = np.array([node_index_dict[v] for u, v in subgraph_edges])  
    data = np.array([1 for _,_ in subgraph_edges])  
    subgraph = ssp.csc_matrix((data, (rows, cols)), shape=(len(subgraph_nodes), len(subgraph_nodes)))
    labels, enclosing_subgraph_nodes = node_label(subgraph, max_distance=100)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = 1
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)
    #print(pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes)
    return ind, pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes

def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[0][0])), dtype=int)
    
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels
    #print(labels)
    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    #print(enclosing_subgraph_nodes)
    return labels, enclosing_subgraph_nodes
