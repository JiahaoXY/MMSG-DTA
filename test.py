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

from models.model import MMSGDTA
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
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def main():
    print(1)
    parser = argparse.ArgumentParser()
    # Add argument
    parser.add_argument('--dataset', required=True, help='davis/kiba/Metz')
    parser.add_argument('--model_path', required=True, type=str, help='model path ready to load')
    args = parser.parse_args()
    params = dict(
        dataset=args.dataset,
        model_path=args.model_path,
    )
    result = []
    dataset = params.get("dataset")
    model_file_name = params.get("model_path")
    
    print(dataset,model_file_name)
    device = torch.device("cuda:0")
    model = MMSGDTA().to(device)
    _,test_data = create_dataset(dataset)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False, collate_fn=collate)
    
    if os.path.isfile(model_file_name):
        model.load_state_dict(torch.load(model_file_name,map_location=torch.device('cpu')),strict=False)
        G,P = predicting(model, device, test_loader)
        ret = [mse(G, P), rmse(G, P), get_cindex(G, P),  get_rm2(G, P), pearson(G, P), spearman(G, P)]
        ret = ['davis',"MMSGDTA"]+[round(e,3) for e in ret]
        result += [ret]
        print('dataset,model,mse,rmse,ci,r2s,pearson,spearman')
        print(ret)
    else:
        print('model is not available!')

    if dataset == 'davis':
        with open('results/result_davis_cold.csv','a') as f:
            f.write('dataset,model,mse,rmse,ci,r2s,pearson,spearman\n')
            for ret in result:
                f.write(','.join(map(str,ret)) + '\n')
    elif dataset == 'kiba':
        with open('results/result_kiba_cold.csv','a') as f:
            f.write('dataset,model,mse,rmse,ci,r2s,pearson,spearman\n')
            for ret in result:
                f.write(','.join(map(str,ret)) + '\n')
    elif dataset == 'metz':
        with open('results/result_metz_cold.csv','a') as f:
            f.write('dataset,model,mse,rmse,ci,r2s,pearson,spearman\n')
            for ret in result:
                f.write(','.join(map(str,ret)) + '\n')

if __name__ == "__main__":
    main()
