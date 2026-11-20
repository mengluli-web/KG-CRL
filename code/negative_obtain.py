import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

def obtain_neg_indep(test_pos,all_pos,valid_heads,valid_tails,neg_all,dics1,dics2,dics3):
    neg_edges1 = []
    neg_edges2 = []
    neg_edges3 = []
    edge_type=39
    for i in test_pos:
        print(i)
        ctn_1 = 0
        ctn_2 = 0
        ctn_3 = 0
        while ctn_1 < 1:
            neg_head = np.random.choice(valid_heads)
            neg_tail = np.random.choice(valid_tails)
            if neg_head != neg_tail and [neg_head, neg_tail] not in all_pos and [neg_head, neg_tail, edge_type] not in neg_all: 
                neg_edges1.append([neg_head, neg_tail, edge_type])
                neg_all.append([neg_head, neg_tail, edge_type])
                dics1[str(i[0])+'\t'+str(i[2])]=[neg_head, neg_tail, edge_type]
                ctn_1+=1
        #print('1_end.....')
        while ctn_2 < 1:
            neg_head = np.random.choice(valid_heads)
            neg_tail = i[2]
            #print(neg_head != neg_tail,[neg_head, neg_tail] not in all_pos,[neg_head, neg_tail, edge_type] not in neg_all)
            if neg_head != neg_tail and [neg_head, neg_tail] not in all_pos and [neg_head, neg_tail, edge_type] not in neg_all: 
                neg_edges2.append([neg_head, neg_tail, edge_type])
                neg_all.append([neg_head, neg_tail, edge_type])
                dics2[str(i[0])+'\t'+str(i[2])]=[neg_head, neg_tail, edge_type]
                ctn_2+=1
        #print('2_end.....')
        while ctn_3 < 1:
            neg_head = i[0]
            neg_tail = np.random.choice(valid_tails)
            if neg_head != neg_tail and [neg_head, neg_tail] not in all_pos and [neg_head, neg_tail,edge_type] not in neg_all: 
                neg_edges3.append([neg_head, neg_tail, edge_type])
                neg_all.append([neg_head, neg_tail, edge_type])
                dics3[str(i[0])+'\t'+str(i[2])]=[neg_head, neg_tail, edge_type]
                ctn_3+=1
        #print('3_end.....')
        
    return neg_edges1,neg_edges2,neg_edges3,test_pos,neg_all,dics1,dics2,dics3

def obtain_neg_cv(test_pos,dics1,dics2,dics3):
    neg_edges1 = []
    neg_edges2 = []
    neg_edges3 = []
    for i in test_pos:
        
        if str(i[0])+'\t'+str(i[2]) in dics1.keys(): 
            print(i,'1........')
            neg_edges1.append(dics1[str(i[0])+'\t'+str(i[2])])
        if str(i[0])+'\t'+str(i[2]) in dics2.keys(): 
            print(i,'2........')
            neg_edges2.append(dics2[str(i[0])+'\t'+str(i[2])])
        if str(i[0])+'\t'+str(i[2]) in dics3.keys(): 
            print(i,'3........')
            neg_edges3.append(dics3[str(i[0])+'\t'+str(i[2])])        
        
    return neg_edges1,neg_edges2,neg_edges3


all_pos = np.loadtxt('../data/edges_mda.txt').astype(int)
valid_heads = np.loadtxt('../data/candidate_microbe.txt').astype(int)
valid_tails = np.loadtxt('../data/candidate_disease.txt').astype(int)
train_pos=np.loadtxt('../data/MDA_indep/train.txt').astype(int)
train_pos=train_pos[train_pos[:,1]==39]
test_pos=np.loadtxt('../data/MDA_indep/valid.txt').astype(int)

neg_all=[]
dics1={}
dics2={}
dics3={}

print('Indep_test..............')
neg_edges1,neg_edges2,neg_edges3,test_pos,neg_all,dics1,dics2,dics3=obtain_neg_indep(test_pos,all_pos,valid_heads,valid_tails,neg_all,dics1,dics2,dics3)
np.savetxt('../data/MDA_indep/valid_random_all.txt',neg_edges1)
np.savetxt('../data/MDA_indep/valid_random_microbe.txt',neg_edges2)            
np.savetxt('../data/MDA_indep/valid_random_disease.txt',neg_edges3)

print('Indep_train..............')
neg_edges1,neg_edges2,neg_edges3,test_pos,neg_all,dics1,dics2,dics3=obtain_neg_indep(train_pos,all_pos,valid_heads,valid_tails,neg_all,dics1,dics2,dics3)
np.savetxt('../data/MDA_indep/train_random_all.txt',neg_edges1)
np.savetxt('../data/MDA_indep/train_random_microbe.txt',neg_edges2)            
np.savetxt('../data/MDA_indep/train_random_disease.txt',neg_edges3)

with open('../data/dics_random_all.plk', 'wb') as f:
    pickle.dump(dics1, f)
with open('../data/dics_random_microbe.plk', 'wb') as f:
    pickle.dump(dics2, f)
with open('../data/dics_random_disease.plk', 'wb') as f:
    pickle.dump(dics3, f)


with open('../data/dics_random_all.plk', 'rb') as f:
    dics1 = pickle.load(f)
with open('../data/dics_random_microbe.plk', 'rb') as f:
    dics2 = pickle.load(f)
with open('../data/dics_random_disease.plk', 'rb') as f:
    dics3 = pickle.load(f)    
    
print(dics1,dics2,dics3)    
for iters in range(5):
    
    print('CV_test_'+str(iters)+'..............')
    test_pos=np.loadtxt('../data/MDA_'+str(iters)+'/valid.txt').astype(int)
    neg_edges1,neg_edges2,neg_edges3=obtain_neg_cv(test_pos,dics1,dics2,dics3)
    np.savetxt('../data/MDA_'+str(iters)+'/valid_random_all.txt',neg_edges1)
    np.savetxt('../data/MDA_'+str(iters)+'/valid_random_microbe.txt',neg_edges2)            
    np.savetxt('../data/MDA_'+str(iters)+'/valid_random_disease.txt',neg_edges3)
    
    print('CV_train_'+str(iters)+'..............')
    train_pos=np.loadtxt('../data/MDA_'+str(iters)+'/train.txt').astype(int)
    train_pos=train_pos[train_pos[:,1]==39]
    neg_edges1,neg_edges2,neg_edges3=obtain_neg_cv(train_pos,dics1,dics2,dics3)
    np.savetxt('../data/MDA_'+str(iters)+'/train_random_all.txt',neg_edges1)
    np.savetxt('../data/MDA_'+str(iters)+'/train_random_microbe.txt',neg_edges2)            
    np.savetxt('../data/MDA_'+str(iters)+'/train_random_disease.txt',neg_edges3)
        
   