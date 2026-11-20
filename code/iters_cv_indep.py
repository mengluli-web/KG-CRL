import os
import shutil
import numpy as np

if not os.path.exists('result/'):
    os.mkdir('result/')
# if not os.path.exists('../data/MDA_1'):
#     os.system('python negative_obtain.py')  
for negative in ['random_all','random_disease','random_microbe']:
    if not os.path.exists('result/models_'+negative+'/'):
        os.mkdir('result/models_'+negative+'/')
    if not os.path.exists('result/attention_'+negative+'/'):
        os.mkdir('result/attention_'+negative+'/')
    os.system('python train_indep.py --dataset %s --negative %s'%('MDA_indep',negative))  
    #for iters in range(5):#CUDA_VISIBLE_DEVICES=1 
    #    os.system('python train.py --dataset %s --negative %s --output %s'%('MDA_'+str(iters),negative,'result/result_MDA_'+str(iters)+'_'+negative)) 
        







