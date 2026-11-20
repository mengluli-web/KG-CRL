import statistics
import timeit
import os
import logging
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
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef,roc_curve
import warnings
warnings.filterwarnings('ignore')

def get_aupr(pre,rec):
    pr_value=0.0
    for ii in range(len(rec[:-1])):
        x_r,x_l=rec[ii],rec[ii+1]
        y_t,y_b=pre[ii],pre[ii+1]
        tempo=abs(x_r-x_l)*(y_t+y_b)*0.5
        pr_value+=tempo
    return pr_value

def comput_scores(y_test, y_pred, th=0.5):           
    y_predlabel = [(0. if item < th else 1.) for item in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SPE = tn*1./(tn+fp)
    MCC = matthews_corrcoef(y_test, y_predlabel)
    fpr,tpr,threshold = roc_curve(y_test, y_predlabel)
    sen, spe, pre, f1, mcc, acc, auc, tn, fp, fn, tp = np.array([recall_score(y_test, y_predlabel), SPE, precision_score(y_test, y_predlabel), 
                                                                  f1_score(y_test, y_predlabel), MCC, accuracy_score(y_test, y_predlabel), 
                                                                  roc_auc_score(y_test, y_pred), tn, fp, fn, tp])
    precision,recall,_ =precision_recall_curve(y_test, y_pred)
    aupr=get_aupr(precision,recall)
    return [aupr, auc, f1, acc, sen, spe, pre]
    
class Trainer():
    def __init__(self, params, graph_classifier, train, train_path, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train
        self.train_data_path = train_path

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0
    
    def train_epoch(self):
        total_loss = 0
        correct_o = 0
        all_labels = []
        all_scores = []
        #print(self.train_data,self.train_data_path)
        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=False, 
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
                                
        dataloader_path = DataLoader(self.train_data_path, batch_size=self.params.batch_size, shuffle=False, 
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)                        
        self.graph_classifier.train()
        #model_params = list(self.graph_classifier.parameters())
        pred_all2 = []
        targets_all2 = []
        train_iters=0
        for batch, batch_path in zip(dataloader, dataloader_path):#b_idx, batch in enumerate(dataloader)
            # Input positive and negative graph 
            data_pos, targets_pos, data_neg, targets_neg, data_cor=self.params.move_batch_to_device(batch, self.params.device)
            data_pos_path, _, data_neg_path, _, _=self.params.move_batch_to_device(batch_path, self.params.device)
            
            self.optimizer.zero_grad()
            self.graph_classifier.train()
            
            pred_all,pred_sub,pred_path,pred_sub_path,embeds = self.graph_classifier(data_pos,data_pos[0].batch_num_nodes,data_pos_path,data_pos_path[0].batch_num_nodes, data_neg,data_neg[0].batch_num_nodes,data_neg_path,data_neg_path[0].batch_num_nodes, eval_random=self.params.with_random,trains=True)
             
            targets_all=torch.cat((targets_pos,targets_neg),dim=0).view(-1)
            
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
            
            pred_all2 += F.sigmoid(pred_all).squeeze(1).detach().cpu().tolist()
            targets_all2 += targets_all.tolist()

            loss = loss_sub_path+self.params.alpha*((loss_subo_pathc+loss_subc_patho)/2)+self.params.beta*((loss_sub_o+loss_sub_co+loss_sub_c)/3)+self.params.gamma*((loss_path_o+loss_path_c+loss_path_co)/3)+self.params.omega*loss_contrastive
            print(f'{train_iters},loss_all: {loss}, loss_sub_path: {loss_sub_path+self.params.alpha*((loss_subo_pathc+loss_subc_patho)/2)}, loss_sub: {self.params.beta*((loss_sub_o+loss_sub_co+loss_sub_c)/3)}, loss_path: {self.params.gamma*((loss_path_o+loss_path_c+loss_path_co)/3)}, loss_contra:{self.params.omega*loss_contrastive}')#
        
            train_iters+=1
            loss.backward()            
            self.optimizer.step()
            self.updates_counter += 1
            
            with torch.no_grad():
                total_loss += loss
        return total_loss,comput_scores(targets_all2, np.array(pred_all2)[:,1])
        

    def train(self):
        
        self.reset_training_state()
        cv_trains=[]
        for epoch in range(self.params.num_epochs):
            time_start = time.time()
            train_loss, train_results = self.train_epoch()
            time_elapsed = time.time() - time_start
            print(f'Epoch {epoch} with loss: {train_loss}, train_results: {train_results} in {time_elapsed}')
            cv_trains.append(train_results)
            #result,_,_,_,_,_ = self.valid_evaluator.eval()
            #print(result)            
            if train_results[3] >= self.best_metric:
                torch.save(self.graph_classifier, 'result/models_'+self.params.negative+'/model_'+str(epoch)+'.pth')
                self.best_metric = train_results[3]
                self.not_improved_count = 0
            else:
                self.not_improved_count += 1
                if self.not_improved_count > self.params.early_stop:
                    logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                    break
        np.savetxt('result/kf_trains_'+str(self.params.negative)+'.txt',np.array(cv_trains))  
        
        data=np.loadtxt('result/kf_trains_'+str(self.params.negative)+'.txt')
        best_epoch=data[:,3].tolist().index(max(data[:,3].tolist()))
        model = torch.load('result/models_'+self.params.negative+'/model_'+str(best_epoch)+'.pth').cuda()
        
        torch.save(model, 'result/model_'+str(self.params.negative)+'.pth')
        result,embeds,preds,targets,att_all,graphs_all_sub,graphs_all_path = self.valid_evaluator.eval()
        
        torch.save(att_all[0], 'result/attention_'+self.params.negative+'/edge_attention_subgraph.pt')
        torch.save(att_all[1], 'result/attention_'+self.params.negative+'/node_attention_subgraph.pt')
        torch.save(att_all[2], 'result/attention_'+self.params.negative+'/edge_attention_path.pt')
        torch.save(att_all[3], 'result/attention_'+self.params.negative+'/node_attention_path.pt')
        with open('result/attention_'+self.params.negative+'/graphs_sub.pkl', 'wb') as f:
            pickle.dump(graphs_all_sub, f)
        with open('result/attention_'+self.params.negative+'/graphs_path.pkl', 'wb') as f:
            pickle.dump(graphs_all_path, f)
        #with open('result/attention_'+self.params.negative+'/graphs_path.pkl', 'rb') as f:
        #    loaded_graphs = pickle.load(f)
        
        ##loaded_graphs = dgl.load_graphs('graphs.bin')
        
        cv_results=[result['auc_pr'],result['auc'],result['f1'],result['acc'],result['sen'],result['spe'],result['pre']]
        np.savetxt('result/result_'+self.params.dataset+'_'+self.params.negative+'.txt',np.array(cv_results)) 
        np.savetxt('result/embeds_'+self.params.dataset+'_'+self.params.negative+'.txt',embeds.detach().cpu().numpy()) 
        np.savetxt('result/preds_'+self.params.dataset+'_'+self.params.negative+'.txt',np.array(preds)) 
        np.savetxt('result/labels_'+self.params.dataset+'_'+self.params.negative+'.txt',np.array(targets)) 
        
    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')
