import os
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse
from preprocessing import create_dataset

from models.model import MGNNDTA
from utils_copy import *
from log.train_logger import TrainLogger
from metrics import *

def predicting(model, device, dataloader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(dataloader.dataset)))
    with torch.no_grad():
        for data in dataloader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)

            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels,data_mol.y.view(-1, 1).cpu()), 0)
    
    # 将预测结果保存到文件
    total_labels_np = total_labels.numpy().flatten()
    total_preds_np = total_preds.numpy().flatten()
    
    with open('/data1/xjh2022388536/project/MGNNSDTA/metz_predicted.csv', 'w') as f:
        f.write('Label,Prediction\n')
        for label, pred in zip(total_labels_np, total_preds_np):
            f.write(f'{label},{pred}\n')

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

result = []
dataset = 'metz'
device = torch.device("cuda:0")
model = MGNNDTA().to(device)
_,test_data = create_dataset(dataset)
test_loader = DataLoader(test_data, batch_size=512, shuffle=False, collate_fn=collate)

model_file_name = ''
if os.path.isfile(model_file_name):
    model.load_state_dict(torch.load(model_file_name,map_location=torch.device('cpu')),strict=False)
    G,P = predicting(model, device, test_loader)
    ret = [mse(G, P), rmse(G, P), get_cindex(G, P),  get_rm2(G, P), pearson(G, P), spearman(G, P)]
    ret = ['davis',"MGNNDTA"]+[round(e,3) for e in ret]
    result += [ret]
    print('dataset,model,mse,rmse,ci,r2s,pearson,spearman')
    print(ret)
else:
    print('model is not available!')

if dataset == 'davis':
    with open('results/result_davis.csv','a') as f:
        f.write('dataset,model,mse,rmse,ci,r2s,pearson,spearman\n')
        for ret in result:
            f.write(','.join(map(str,ret)) + '\n')
elif dataset == 'kiba':
    with open('results/result_kiba.csv','a') as f:
        f.write('dataset,model,mse,rmse,ci,r2s,pearson,spearman\n')
        for ret in result:
            f.write(','.join(map(str,ret)) + '\n')
elif dataset == 'metz':
    with open('results/result_metz.csv','a') as f:
        f.write('dataset,model,mse,rmse,ci,r2s,pearson,spearman\n')
        for ret in result:
            f.write(','.join(map(str,ret)) + '\n')


