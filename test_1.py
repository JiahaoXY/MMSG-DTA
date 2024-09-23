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
import pandas as pd
from models.model import MGNNDTA
from utils_copy import *
from log.train_logger import TrainLogger
from metrics import *

def predicting(model, device, dataloader):
    model.eval()
    total_preds = torch.Tensor()
    #total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(dataloader.dataset)))
    with torch.no_grad():
        for data in dataloader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)

            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            #total_labels = torch.cat((total_labels,data_mol.y.view(-1, 1).cpu()), 0)
    
    # 读取现有的CSV文件
    # 将预测结果转换为NumPy格式
    total_preds_np = total_preds.numpy().flatten()
    csv_file_path = '/data1/xjh2022388536/project/MGNNSDTA/EGCR_aff.csv'
    df = pd.read_csv(csv_file_path)

    # 检查行数是否匹配
    if len(total_preds_np) != len(df):
        raise ValueError(f"预测值数量({len(total_preds_np)})与CSV文件中的行数({len(df)})不匹配！")

    # 将预测值作为新的一列添加到DataFrame
    df['Prediction'] = total_preds_np

    # 保存修改后的DataFrame到CSV文件
    df.to_csv(csv_file_path, index=False)
    
    return total_preds.numpy().flatten()

result = []
dataset = 'EGFR'
device = torch.device("cuda:1")
model = MGNNDTA().to(device)
test_data = create_dataset(dataset)

test_loader = DataLoader(test_data, batch_size=256, shuffle=False, collate_fn=collate)

model_file_name = '/data1/xjh2022388536/project/MGNNSDTA/results/all/20240805_185229_kiba/model/KIBA_all_V2.pt'
if os.path.isfile(model_file_name):
    model.load_state_dict(torch.load(model_file_name,map_location=torch.device('cpu')),strict=False)
    P = predicting(model, device, test_loader)
#     ret = [mse(G, P), rmse(G, P), get_cindex(G, P),  get_rm2(G, P), pearson(G, P), spearman(G, P)]
#     ret = ['davis',"MGNNDTA"]+[round(e,3) for e in ret]
#     result += [ret]
#     print('dataset,model,mse,rmse,ci,r2s,pearson,spearman')
#     print(ret)
# else:
#     print('model is not available!')

# if dataset == 'davis':
#     with open('results/result_davis.csv','a') as f:
#         f.write('dataset,model,mse,rmse,ci,r2s,pearson,spearman\n')
#         for ret in result:
#             f.write(','.join(map(str,ret)) + '\n')
# elif dataset == 'kiba':
#     with open('results/result_kiba.csv','a') as f:
#         f.write('dataset,model,mse,rmse,ci,r2s,pearson,spearman\n')
#         for ret in result:
#             f.write(','.join(map(str,ret)) + '\n')
# elif dataset == 'metz':
#     with open('results/result_metz.csv','a') as f:
#         f.write('dataset,model,mse,rmse,ci,r2s,pearson,spearman\n')
#         for ret in result:
#             f.write(','.join(map(str,ret)) + '\n')


