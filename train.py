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
from metrics import *

from models.model import MGNNDTA
from utils_copy import *
from log.train_logger import TrainLogger


def main():
    parser = argparse.ArgumentParser()
    # Add argument
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()

    params = dict(
        save_dir="/results/all",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    BATCH_SZIE = params.get("batch_size")
    LR = params.get('lr')
    train_data , test_data = create_dataset(DATASET)
    train_loader = DataLoader(train_data, batch_size=BATCH_SZIE, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=BATCH_SZIE, shuffle=False,  collate_fn=collate)

    device = torch.device("cuda:0")
    model = MGNNDTA().to(device)

    epochs = 3000
    best_mse = 1000
    best_epoch =  -1 

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        print('Training on {} samples...'.format(len(train_loader.dataset)))
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)

            output = model(data_mol, data_pro)
            loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
            loss.backward()
            optimizer.step()
            # running_loss.update(loss.item(), data[1].y.size(0))
        print('Train epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))
        model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
        with torch.no_grad():
            for data in test_loader:
                data_mol = data[0].to(device)
                data_pro = data[1].to(device)

                output = model(data_mol, data_pro)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels,data_mol.y.view(-1, 1).cpu()), 0)
        G = total_labels.numpy().flatten()
        P = total_preds.numpy().flatten()

        test_loss = mse(G,P)

        msg =  "epoch-%d, mse-%.4f"%(epoch+1,test_loss)
        logger.info(msg)
        if test_loss<best_mse:
            save_model_dict(model, logger.get_model_dir(), msg)
            best_epoch = epoch+1
            best_mse = test_loss
            print('rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,'MGNNDTA',DATASET)
        else:
            print('No improvement since epoch ', best_epoch, '; best_mse:', best_mse,'MGNNDTA',DATASET)
    print('train success!')

if __name__ == "__main__":
    main()
