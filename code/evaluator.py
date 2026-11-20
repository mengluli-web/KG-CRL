import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

class Evaluator():
    def __init__(self, params, graph_classifier, data, data_path):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data
        self.data_path = data_path
    def eval(self, save=True):
        pred_all2 = []
        targets_all2 = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        dataloader_path = DataLoader(self.data_path, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)    

        self.graph_classifier.eval()
        with torch.no_grad():
            for batch, batch_path in zip(dataloader, dataloader_path):#b_idx, batch in enumerate(dataloader)
                data_pos, targets_pos, data_neg, targets_neg, data_cor=self.params.move_batch_to_device(batch, self.params.device)
                data_pos_path, _, data_neg_path, _, _=self.params.move_batch_to_device(batch_path, self.params.device)
                
                pred_all = self.graph_classifier(data_pos,data_pos[0].batch_num_nodes,data_pos_path,data_pos_path[0].batch_num_nodes,data_neg,data_neg[0].batch_num_nodes,data_neg_path,data_neg_path[0].batch_num_nodes, eval_random=self.params.eval_random,trains=False)
                targets_all=torch.cat((targets_pos,targets_neg),dim=0).view(-1)            
                pred_all2 += pred_all.squeeze(1).detach().cpu().tolist()
                targets_all2 += targets_all.tolist()
                #print(comput_scores(targets_all2, np.array(pred_all2)[:,1]))
    
        assert not torch.any(torch.isnan(torch.tensor(targets_all)))
        auc_pr, auc, f1, acc, sen, spe, pre=comput_scores(targets_all2, np.array(pred_all2)[:,1])

        return {'auc': auc, 'auc_pr': auc_pr,'f1':f1,'acc':acc, 'sen':sen, 'spe':spe, 'pre':pre}
