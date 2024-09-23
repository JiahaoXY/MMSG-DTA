import torch
import esm
import math
import numpy as np

import json, pickle
from collections import OrderedDict
import os
from tqdm import tqdm

# data prepare
def protein_graph_construct(proteins, save_dir):
    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    target_graph = {}
    count = 0
    key_list=[]
    for key in proteins:
        key_list.append(key)

    for k_i in tqdm(range(len(key_list))):
        key=key_list[k_i]
        data = []
        pro_id = key
        seq = proteins[key]
        if len(seq) <= 1200:
            data.append((pro_id, seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            contact_map = results["contacts"][0].numpy()
            target_graph[pro_id] = contact_map
        else:
            contact_prob_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
            interval = 500
            i = math.ceil(len(seq) / interval)
            # ======================
            # =                    =
            # =                    =
            # =          ======================
            # =          =*********=          =
            # =          =*********=          =
            # ======================          =
            #            =                    =
            #            =                    =
            #            ======================
            # where * is the overlapping area
            # subsection seq contact map prediction
            for s in range(i):
                start = s * interval  # sub seq predict start
                end = min((s + 2) * interval, len(seq))  # sub seq predict end
                sub_seq_len = end - start
                # prediction
                temp_seq = seq[start:end]
                temp_data = []
                temp_data.append((pro_id, temp_seq))
                batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                # insert into the global contact map
                row, col = np.where(contact_prob_map[start:end, start:end] != 0)
                row = row + start
                col = col + start
                contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][
                    0].numpy()
                contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0
                if end == len(seq):
                    break
            target_graph[pro_id] = contact_prob_map
        np.save(save_dir + pro_id + '.npy', target_graph[pro_id])
        count += 1

if __name__ == '__main__':
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    # # for test
    dataset = 'EGFR'
    if dataset in ['kiba', 'davis','Metz','EGFR']:
        #proteins = json.load(open('dataset/' + dataset + '/proteins.txt'), object_pairs_hook=OrderedDict)
        proteins = json.load(open('dataset/' + dataset + '/proteins.txt'))
    print('dataset:', dataset)
    print(len(proteins))

    save_dir = 'dataset/' + dataset + '/pconsc4/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    protein_graph_construct(proteins, save_dir)
