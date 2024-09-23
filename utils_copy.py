import os
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset, DataLoader, Batch

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)

def load_checkpoint(model_path):
    return torch.load(model_path)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x


VOCAB_PROTEIN = { "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, 
				"H": 7, "I": 8, "K": 9, "L": 10, "M": 11, "N": 12, 
				"P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, 
				"W": 19, "Y": 20, "X": 21}

VOCAB_LIGAND = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
def PROTEIN2INT(target):
    return [VOCAB_PROTEIN[s] for s in target] 

def MOL2INT(smi):
    return [VOCAB_LIGAND[s] for s in smi]
# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='data', dataset='davis',
                drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None, transform=None,
                pre_transform=None, smile_graph=None,  target_graph=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.drug_smiles = drug_smiles
        self.target_sequence = target_sequence
        self.y = y
        self.smile_graph = smile_graph
        self.target_graph = target_graph

        self.process(drug_smiles, target_sequence, x, x_mask, xt, xt_mask, y, smile_graph, target_graph)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None, smile_graph=None, target_graph=None):
        assert (len(drug_smiles) == len(target_sequence) and len(drug_smiles) == len(y)), 'The three lists must have the same length!'
        data_list_mol = []
        data_list_pro = []

        data_len = len(drug_smiles)
        print('loading tensors ...')
        for i in tqdm(range(data_len)):

            smiles = drug_smiles[i]
            tar_seq = target_sequence[i]
            if x is not None:
                drug = x[i]
            if x_mask is not None:
                drug_mask = x_mask[i]
            if xt is not None:
                target = xt[i]
            if xt_mask is not None:
                target_mask = xt_mask[i]
            labels = y[i]
            mol_size, mol_features, mol_edge_index, mol_edge_attr= smile_graph[smiles]
            target_size, target_features, target_edge_index, target_edge_weight = target_graph[tar_seq]

            drug_seq = MOL2INT(smiles)
            drug_seq_len = 220
            if len(drug_seq) < drug_seq_len:
                mol_seq_emb = np.pad(drug_seq, (0, drug_seq_len- len(drug_seq)))
            else:
                mol_seq_emb = drug_seq[:drug_seq_len]
            GCNData_mol = DATA.Data(x=mol_features,
                                    edge_index=mol_edge_index,
                                    edge_attr = mol_edge_attr,
                                    mol_emb = torch.LongTensor([mol_seq_emb]),
                                    y=torch.FloatTensor([labels]))
            if x is not None:
                GCNData_mol.drug = torch.LongTensor([drug])
            if x_mask is not None:
                GCNData_mol.drug_mask = torch.LongTensor([drug_mask])
            GCNData_mol.__setitem__('c_size', torch.LongTensor([mol_size]))

            target_seq = PROTEIN2INT(tar_seq)
            target_seq_len = 1200
            if len(target_seq) < target_seq_len:
                pro_seq_emb = np.pad(target_seq, (0, target_seq_len- len(target_seq)))
            else:
                pro_seq_emb = target_seq[:target_seq_len]

            GCNData_pro = DATA.Data(x=target_features,
                                    edge_index=target_edge_index,
                                    edge_attr = target_edge_weight,
                                    pro_emb = torch.LongTensor([pro_seq_emb]),
                                    y=torch.FloatTensor([labels]))
            
            if xt is not None:
                GCNData_pro.target = torch.LongTensor([target])
            if xt_mask is not None:
                GCNData_pro.target_mask = torch.LongTensor([target_mask])
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)


        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]


        self.data_mol = data_list_mol
        self.data_pro = data_list_pro


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return GNNData_mol, GNNData_pro
        return self.data_mol[idx], self.data_pro[idx]

def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB



