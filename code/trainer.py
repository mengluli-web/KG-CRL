import statistics
import timeit
import os
import pdb
import numpy as np
import time
from data_utils import process_files, save_to_file
from graph_utils import ssp_multigraph_to_dgl, incidence_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import dgl
from sklearn import metrics
import pickle

class Trainer():
    def __init__(self, params, graph_classifier, train, train_path, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train
        self.train_data_path = train_path
        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        print('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0
    
    def train_epoch(self):
        total_loss = 0
        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=False, 
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
                                
        dataloader_path = DataLoader(self.train_data_path, batch_size=self.params.batch_size, shuffle=False, 
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)                        
        self.graph_classifier.train()
        #model_params = list(self.graph_classifier.parameters())
        train_iters=0
        
        for batch, batch_path in zip(dataloader, dataloader_path):#b_idx, batch in enumerate(dataloader)
            # Input positive and negative graph 
            
            data_pos, targets_pos, data_neg, targets_neg, data_cor=self.params.move_batch_to_device(batch, self.params.device)
            data_pos_path, _, data_neg_path, _, _=self.params.move_batch_to_device(batch_path, self.params.device)
            
            targets_all=torch.cat((targets_pos,targets_neg),dim=0).view(-1)
            #print(node_pair_pos,node_pair_neg)
            self.optimizer.zero_grad()
            self.graph_classifier.train()
            
            if self.params.crl=='with_crl' and self.params.sub=='with_subgraph' and self.params.path=='with_path' and self.params.CLoss=='with_CLoss':
                pred_all,pred_sub,pred_path,pred_sub_path,embeds = self.graph_classifier(data_pos,data_pos[0].batch_num_nodes,data_pos_path,data_pos_path[0].batch_num_nodes, data_neg,data_neg[0].batch_num_nodes,data_neg_path,data_neg_path[0].batch_num_nodes, eval_random=self.params.with_random,trains=True)
                
                
                dis_sub_pos_neg = F.pairwise_distance(embeds[0], embeds[2])
                dis_path_pos_neg = F.pairwise_distance(embeds[1], embeds[3])
                loss_contrastive = torch.mean(torch.pow(torch.clamp(2 - dis_sub_pos_neg, min=0.0), 2)+\
                                                torch.pow(torch.clamp(2 - dis_path_pos_neg, min=0.0), 2))
                '''
                dis_sub_path_pos = F.pairwise_distance(embeds[0], embeds[1])
                dis_sub_path_neg = F.pairwise_distance(embeds[2], embeds[3])
                dis_sub_pos_neg = F.pairwise_distance(embeds[0], embeds[2])
                dis_path_pos_neg = F.pairwise_distance(embeds[1], embeds[3])
                dis_sub_path_pos_neg = F.pairwise_distance(embeds[0], embeds[3])
                dis_sub_path_pos_neg2 = F.pairwise_distance(embeds[1], embeds[2])
                loss_contrastive = torch.mean(torch.pow(dis_sub_path_pos, 2) + torch.pow(dis_sub_path_neg, 2) +
                                      torch.pow(torch.clamp(1 - dis_sub_pos_neg, min=0.0), 2)+
                                      torch.pow(torch.clamp(1 - dis_path_pos_neg, min=0.0), 2)+
                                      torch.pow(torch.clamp(1 - dis_sub_path_pos_neg, min=0.0), 2)+
                                      torch.pow(torch.clamp(1 - dis_sub_path_pos_neg2, min=0.0), 2))                             
                '''
                
                loss_sub_co = nn.CrossEntropyLoss()(pred_sub[2], targets_all)
                loss_sub_c = F.kl_div(pred_sub[0], torch.ones_like(pred_sub[0], dtype=torch.float).to(self.params.device) / self.params.num_classes, reduction='batchmean')
                loss_path_co = nn.CrossEntropyLoss()(pred_path[2], targets_all)
                loss_path_c = F.kl_div(pred_path[0], torch.ones_like(pred_path[0], dtype=torch.float).to(self.params.device) / self.params.num_classes, reduction='batchmean')
                loss_subo_pathc = nn.CrossEntropyLoss()(pred_sub_path[0], targets_all)
                loss_subc_patho = nn.CrossEntropyLoss()(pred_sub_path[1], targets_all)
                
                loss_sub_o = nn.CrossEntropyLoss()(pred_sub[1], targets_all)
                loss_path_o = nn.CrossEntropyLoss()(pred_path[1], targets_all)
                loss_sub_path = nn.CrossEntropyLoss()(pred_all, targets_all)#F.nll_loss(pred_all, targets_all)
                
                loss = loss_sub_path+self.params.alpha*((loss_subo_pathc+loss_subc_patho)/2)+self.params.beta*((loss_sub_o+loss_sub_co+loss_sub_c)/3)+self.params.gamma*((loss_path_o+loss_path_c+loss_path_co)/3)+self.params.omega*loss_contrastive
                print(f'{train_iters},loss_all: {loss}, loss_sub_path: {loss_sub_path+self.params.alpha*((loss_subo_pathc+loss_subc_patho)/2)}, loss_sub: {self.params.beta*((loss_sub_o+loss_sub_co+loss_sub_c)/3)}, loss_path: {self.params.gamma*((loss_path_o+loss_path_c+loss_path_co)/3)}, loss_contra:{self.params.omega*loss_contrastive}')#
                '''
                
                loss_sub_path = nn.CrossEntropyLoss()(pred_all, targets_all)
                loss_sub_o = nn.CrossEntropyLoss()(pred_sub[1], targets_all)
                loss_path_o = nn.CrossEntropyLoss()(pred_path[1], targets_all)
                loss = loss_sub_path+self.params.beta*loss_sub_o+self.params.gamma*loss_path_o+self.params.omega*loss_contrastive
                print(f'{train_iters},loss_all: {loss}, loss_sub_path: {loss_sub_path}, loss_sub: {self.params.beta*loss_sub_o}, loss_path: {self.params.gamma*loss_path_o}, loss_contra:{self.params.omega*loss_contrastive}')
                '''
                
            elif self.params.crl=='without_crl':
                pred_all,pred_sub,pred_path,pred_sub_path,embeds = self.graph_classifier(data_pos,data_pos[0].batch_num_nodes,data_pos_path,data_pos_path[0].batch_num_nodes, data_neg,data_neg[0].batch_num_nodes,data_neg_path,data_neg_path[0].batch_num_nodes, eval_random=self.params.with_random,trains=True)
                
                '''
                dis_sub_pos_neg = F.pairwise_distance(embeds[0], embeds[2])
                dis_path_pos_neg = F.pairwise_distance(embeds[1], embeds[3])
                loss_contrastive = torch.mean(torch.pow(torch.clamp(2 - dis_sub_pos_neg, min=0.0), 2)+\
                                                torch.pow(torch.clamp(2 - dis_path_pos_neg, min=0.0), 2))
                '''
                dis_sub_path_pos = F.pairwise_distance(embeds[0], embeds[1])
                dis_sub_path_neg = F.pairwise_distance(embeds[2], embeds[3])
                dis_sub_pos_neg = F.pairwise_distance(embeds[0], embeds[2])
                dis_path_pos_neg = F.pairwise_distance(embeds[1], embeds[3])
                dis_sub_path_pos_neg = F.pairwise_distance(embeds[0], embeds[3])
                dis_sub_path_pos_neg2 = F.pairwise_distance(embeds[1], embeds[2])
                loss_contrastive = torch.mean(torch.pow(dis_sub_path_pos, 2) + torch.pow(dis_sub_path_neg, 2) +
                                      torch.pow(torch.clamp(1 - dis_sub_pos_neg, min=0.0), 2)+
                                      torch.pow(torch.clamp(1 - dis_path_pos_neg, min=0.0), 2)+
                                      torch.pow(torch.clamp(1 - dis_sub_path_pos_neg, min=0.0), 2)+
                                      torch.pow(torch.clamp(1 - dis_sub_path_pos_neg2, min=0.0), 2))    
            
                loss_sub = nn.CrossEntropyLoss()(pred_sub[1], targets_all)
                loss_path = nn.CrossEntropyLoss()(pred_path[1], targets_all)
                loss_sub_path = nn.CrossEntropyLoss()(pred_all, targets_all)
                loss = loss_sub_path+self.params.beta*loss_sub+self.params.gamma*loss_path+self.params.omega*loss_contrastive
                print(f'{train_iters},loss_all: {loss}, loss_sub_path: {loss_sub_path}, loss_sub: {self.params.beta*loss_sub}, loss_path: {self.params.gamma*loss_path}, loss_contra:{self.params.omega*loss_contrastive}')
                    
            elif self.params.sub=='without_subgraph':
                pred_path,embeds = self.graph_classifier(data_pos,data_pos[0].batch_num_nodes,data_pos_path,data_pos_path[0].batch_num_nodes, data_neg,data_neg[0].batch_num_nodes,data_neg_path,data_neg_path[0].batch_num_nodes, eval_random=self.params.with_random,trains=True)
                  
                dis_path_pos_neg = F.pairwise_distance(embeds[0], embeds[1])
                loss_contrastive = torch.mean(torch.pow(torch.clamp(1 - dis_path_pos_neg, min=0.0), 2))
                
                loss_path_co = nn.CrossEntropyLoss()(pred_path[2], targets_all)
                loss_path_c = F.kl_div(pred_path[0], torch.ones_like(pred_path[0], dtype=torch.float).to(self.params.device) / self.params.num_classes, reduction='batchmean')
                loss_path_o = nn.CrossEntropyLoss()(pred_path[1], targets_all)
                
                loss = loss_path_o+loss_path_c+loss_path_co+self.params.omega*loss_contrastive

                print(f'{train_iters},loss_all: {loss}, loss_path: {loss_path_o}, loss_path: {loss_path_c+loss_path_co}, loss_contra:{self.params.omega*loss_contrastive}')
                
            elif self.params.path=='without_path':
                pred_sub,embeds = self.graph_classifier(data_pos,data_pos[0].batch_num_nodes,data_pos_path,data_pos_path[0].batch_num_nodes, data_neg,data_neg[0].batch_num_nodes,data_neg_path,data_neg_path[0].batch_num_nodes, eval_random=self.params.with_random,trains=True)
                
                dis_sub_pos_neg = F.pairwise_distance(embeds[0], embeds[1])
                loss_contrastive = torch.mean(torch.pow(torch.clamp(1 - dis_sub_pos_neg, min=0.0), 2))
                
                loss_sub_co = nn.CrossEntropyLoss()(pred_sub[2], targets_all)
                loss_sub_c = F.kl_div(pred_sub[0], torch.ones_like(pred_sub[0], dtype=torch.float).to(self.params.device) / self.params.num_classes, reduction='batchmean')
                
                loss_sub_o = nn.CrossEntropyLoss()(pred_sub[1], targets_all)
                
                loss = loss_sub_o+loss_sub_co+loss_sub_c+self.params.omega*loss_contrastive
                print(f'{train_iters},loss_all: {loss}, loss_sub: {loss_sub_o}, loss_sub: {loss_sub_co+loss_sub_c}, loss_contra:{self.params.omega*loss_contrastive}')
            
            elif self.params.CLoss=='without_CLoss':
                pred_all,pred_sub,pred_path,pred_sub_path,embeds = self.graph_classifier(data_pos,data_pos[0].batch_num_nodes,data_pos_path,data_pos_path[0].batch_num_nodes, data_neg,data_neg[0].batch_num_nodes,data_neg_path,data_neg_path[0].batch_num_nodes, eval_random=self.params.with_random,trains=True)
               
                loss_sub_co = nn.CrossEntropyLoss()(pred_sub[2], targets_all)
                loss_sub_c = F.kl_div(pred_sub[0], torch.ones_like(pred_sub[0], dtype=torch.float).to(self.params.device) / self.params.num_classes, reduction='batchmean')
                loss_path_co = nn.CrossEntropyLoss()(pred_path[2], targets_all)
                loss_path_c = F.kl_div(pred_path[0], torch.ones_like(pred_path[0], dtype=torch.float).to(self.params.device) / self.params.num_classes, reduction='batchmean')
                loss_subo_pathc = nn.CrossEntropyLoss()(pred_sub_path[0], targets_all)
                loss_subc_patho = nn.CrossEntropyLoss()(pred_sub_path[1], targets_all)
                
                loss_sub_o = nn.CrossEntropyLoss()(pred_sub[1], targets_all)
                loss_path_o = nn.CrossEntropyLoss()(pred_path[1], targets_all)
                loss_sub_path = nn.CrossEntropyLoss()(pred_all, targets_all)#F.nll_loss(pred_all, targets_all)
                
                loss = loss_sub_path+self.params.alpha*((loss_subo_pathc+loss_subc_patho)/2)+self.params.beta*((loss_sub_o+loss_sub_co+loss_sub_c)/3)+self.params.gamma*((loss_path_o+loss_path_c+loss_path_co)/3)
                print(f'{train_iters},loss_all: {loss}, loss_sub_path: {loss_sub_path+self.params.alpha*((loss_subo_pathc+loss_subc_patho)/2)}, loss_sub: {self.params.beta*((loss_sub_o+loss_sub_co+loss_sub_c)/3)}, loss_path: {self.params.gamma*((loss_path_o+loss_path_c+loss_path_co)/3)}')#
            
            train_iters+=1
            loss.backward()            
            self.optimizer.step()
            self.updates_counter += 1
            
            with torch.no_grad():
                total_loss += loss
        return total_loss

    def train(self):
        self.reset_training_state()
        cv_results=[]
        best_results=[]
        best_epoch=0
        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            train_loss = self.train_epoch()
            time_elapsed = time.time() - time_start
            print(f'Epoch {epoch} with loss: {train_loss} in {time_elapsed}')
                        
            tic = time.time()
            result = self.valid_evaluator.eval()
            print('\nPerformance:' + str(result) + 'in ' + str(time.time() - tic))

            if result['acc'] >= self.best_metric:
                self.best_metric = result['acc']
                best_results = result
                best_epoch = epoch
                self.not_improved_count = 0
            else:
                self.not_improved_count += 1
                if self.not_improved_count > self.params.early_stop:
                    print(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                    break
            cv_results.append([result['auc_pr'],result['auc'],result['f1'],result['acc'],result['sen'],result['spe'],result['pre']])
            print(best_epoch,best_results)
        np.savetxt(self.params.output+'.txt',np.array(cv_results)) 

