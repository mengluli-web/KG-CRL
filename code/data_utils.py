import os
import pdb
import logging
import numpy as np
from scipy.sparse import csc_matrix

def process_files(files, node_type1,node_type2):
    entity2id = {}
    relation2id = {}# if saved_relation2id is None else saved_relation2id
    triplets = {}
    for file_type, file_path in files.items():
        data=[]
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            entity2id[triplet[0]] = int(triplet[0])
            entity2id[triplet[2]] = int(triplet[2])
            relation2id[triplet[1]] = int(triplet[1])
            data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)
    
    candidate_microbe = np.loadtxt('../data/candidate_'+node_type1+'.txt').astype(int)
    candidate_disease = np.loadtxt('../data/candidate_'+node_type2+'.txt').astype(int)
    for i in candidate_microbe:
        entity2id[str(i)] = i
    for i in candidate_disease:
        entity2id[str(i)] = i
    
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    
    # Construct the the neighbor relations of each entity
    num_rels = len(id2relation)
    num_ents = len(entity2id)
    h2r = {}
    h2r_len = {}
    t2r = {}
    t2r_len = {}
    
    for triplet in triplets['train']:
        h, t, r = triplet
        if h not in h2r:
            h2r_len[h] = 1
            h2r[h] = [r]
        else:
            h2r_len[h] += 1
            h2r[h].append(r)
        if t not in t2r:
            t2r[t] = [r]
            t2r_len[t]  = 1
        else:
            t2r[t].append(r)
            t2r_len[t] += 1
    
    h_nei_rels_len = int(np.percentile(list(h2r_len.values()), 75))
    t_nei_rels_len = int(np.percentile(list(t2r_len.values()), 75))
    # logging.info("Average number of relations each node: ", "head: ", h_nei_rels_len, 'tail: ', t_nei_rels_len)
    
    # The index "num_rels" of relation is considered as "padding" relation.
    # Use padding relation to initialize matrix of ent2rels.
    m_h2r = np.ones([num_ents, h_nei_rels_len]) * num_rels
    for ent, rels in h2r.items():
        if len(rels) > h_nei_rels_len:
            rels = np.array(rels)[np.random.choice(np.arange(len(rels)), h_nei_rels_len)]
            m_h2r[ent] = rels
        else:
            rels = np.array(rels)
            m_h2r[ent][: rels.shape[0]] = rels      
    
    m_t2r = np.ones([num_ents, t_nei_rels_len]) * num_rels
    for ent, rels in t2r.items():
        if len(rels) > t_nei_rels_len:
            rels = np.array(rels)[np.random.choice(np.arange(len(rels)), t_nei_rels_len)]
            m_t2r[ent] = rels
        else:
            rels = np.array(rels)
            m_t2r[ent][: rels.shape[0]] = rels      
    
    print("Construct matrix of ent2rels done!")
    
    triplets['train'] = triplets['train'][np.argsort(triplets['train'][:,2])]
        
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, m_h2r, m_t2r


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')
