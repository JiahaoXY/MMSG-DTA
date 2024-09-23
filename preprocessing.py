import os
import torch
import os.path as osp
import json, pickle
import numpy as np
import pandas as pd
import networkx as nx

from utils import *
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDConfig
from collections import OrderedDict
from rdkit.Chem import ChemicalFeatures
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

VOCAB_PROTEIN = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                'X']

def read_data(filename):
    df = pd.read_csv('dataset/'+filename)
    drugs, prots= list(df['compound_iso_smiles']),list(df['target_sequence'])
    return drugs, prots

def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

seq_dict = {v:(i+1) for i,v in enumerate(VOCAB_PROTEIN)}
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])

# nomarlize the residue feature
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def encoding_unk(x, allowable_set):
    list = [False for i in range(len(allowable_set))]
    i = 0
    for atom in x:
        if atom in allowable_set:
            list[allowable_set.index(atom)] = True
            i += 1
    if i != len(x):
        list[-1] = True
    return list

def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]

    return np.array(res_property1 + res_property2)

# target feature for target graph
def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(VOCAB_PROTEIN), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in VOCAB_PROTEIN:
                    count += 1
                    continue
                pfm_mat[VOCAB_PROTEIN.index(res), count] += 1
                count += 1

    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat

    return pssm_mat

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(VOCAB_PROTEIN)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_hot_encoding(pro_seq[i], VOCAB_PROTEIN)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)

    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

def target2feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    feature = target_feature(aln_file, target_sequence)

    return feature


# target sequence to target graph
def sequence2graph(target_key, target_sequence, distance_dir):
    target_edge_index = []
    target_edge_distance = []
    target_size = len(target_sequence)

    target_feature = seq_feature(target_sequence)
    contact_map_file = os.path.join(distance_dir, target_key + '.npy')
    distance_map = np.load(contact_map_file)

    for i in range(target_size):
        distance_map[i, i] = 1
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1
    index_row, index_col = np.where(distance_map >= 0.5)  # for threshold

    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])  # dege
        target_edge_distance.append(distance_map[i, j])  # edge weight

    target_feature = torch.Tensor(target_feature)
    target_edge_index = torch.LongTensor(target_edge_index).transpose(1, 0)
    target_edge_distance = torch.FloatTensor(target_edge_distance)

    return target_size, target_feature, target_edge_index, target_edge_distance

def get_nodes(g):
    feat = []
    for n, d in g.nodes(data=True):
        h_t = []
        h_t += [int(d['a_type'] == x) for x in ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                                'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                                'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li',
                                                'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                                'Pt', 'Hg', 'Pb', 'X']]
        h_t.append(d['a_num'])
        h_t.append(d['acceptor'])
        h_t.append(d['donor'])
        h_t.append(int(d['aromatic']))
        h_t += [int(d['degree'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['ImplicitValence'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['num_h'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['hybridization'] == x) for x in (Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3)]
        # 5 more
        h_t.append(d['ExplicitValence'])
        h_t.append(d['FormalCharge'])
        h_t.append(d['NumExplicitHs'])
        h_t.append(d['NumRadicalElectrons'])
        feat.append((n, h_t))
    feat.sort(key=lambda item: item[0])
    node_attr = torch.FloatTensor([item[1] for item in feat])
    return node_attr

def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
                for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]

        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t

    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))


    return edge_index, edge_attr

# mol smile to mol graph edge index
def smile2graph(smile):
    mol = Chem.MolFromSmiles(smile)

    feats = chem_feature_factory.GetFeaturesForMol(mol)
    mol_size = mol.GetNumAtoms()
    g = nx.DiGraph()
    
    # Create nodes
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        g.add_node(i,
                a_type=atom_i.GetSymbol(),
                a_num=atom_i.GetAtomicNum(),
                acceptor=0,
                donor=0,
                aromatic=atom_i.GetIsAromatic(),
                hybridization=atom_i.GetHybridization(),
                num_h=atom_i.GetTotalNumHs(),
                degree = atom_i.GetDegree(),
                # 5 more node features
                ExplicitValence=atom_i.GetExplicitValence(),
                FormalCharge=atom_i.GetFormalCharge(),
                ImplicitValence=atom_i.GetImplicitValence(),
                NumExplicitHs=atom_i.GetNumExplicitHs(),
                NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
            )
            
    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['acceptor']
    # Read Edges
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                            b_type=e_ij.GetBondType(),
                            # 1 more edge features 2 dim
                            IsConjugated=int(e_ij.GetIsConjugated()),
                            )
                
    node_attr = get_nodes(g)
    edge_index, edge_attr = get_edges(g)         

    return mol_size, node_attr, edge_index, edge_attr

def create_dataset(dataset):
    dataset_dir = os.path.join('/dataset',dataset)
    # drug smiles
    ligands = json.load(open(os.path.join(dataset_dir, 'ligands_can.txt')), object_pairs_hook=OrderedDict)
    # protein sequences
    proteins = json.load(open(os.path.join(dataset_dir, 'proteins.txt')), object_pairs_hook=OrderedDict)

    # load protein feature and predicted distance map
    process_dir = os.path.join('dataset/')
    pro_distance_dir = os.path.join(process_dir, dataset, 'pconsc4')  # numpy .npy file
    pro_msa_path = os.path.join(process_dir, dataset, 'aln')  # numpy .npy file

    # dataset process
    drugs = []  # rdkit entity
    prots = []  # sequences
    prot_keys = []  # protein id (or name)
    drug_smiles = []  # smiles
    # create molecule graph
    print("create molecule graph ...")
    # smiles
    for d in ligands.keys():
        # if dataset == 'metz':
        #     lg = ligands[d]
        # else:
        #     lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        lg = ligands[d]
        drugs.append(lg)
        drug_smiles.append(ligands[d])
        smile_graph = {}

    for i in tqdm(range(len(drugs))):
        smile = drugs[i]
        g_d = smile2graph(smile)
        smile_graph[smile] = g_d

    print("create protein graph ...")  
    # seqs
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)  

    target_graph = {}
    for i in tqdm(range(len(prot_keys))):
        key = prot_keys[i]
        protein = prots[i]
        g_t = sequence2graph(key, protein, pro_distance_dir)
        target_graph[protein] = g_t
    
    # read files(train and test)
    #train_csv = dataset + '/raw/data_train.csv'
    test_csv = dataset + '/raw/data_test.csv'
    #train_drugs, train_prots, train_Y = read_data(train_csv)
    test_drugs, test_prots = read_data(test_csv)
    print(len(test_drugs),len(test_prots))
    #train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
    test_drugs, test_prots = np.asarray(test_drugs), np.asarray(test_prots)
    
    #train_data = DTADataset(root='dataset', dataset=dataset + '_' + 'train', drug_smiles=train_drugs, target_sequence=train_prots, y=train_Y,
                            #smile_graph=smile_graph,target_graph=target_graph)
    
    test_data = DTADataset(root='dataset', dataset=dataset + '_' + 'test', drug_smiles=test_drugs, target_sequence=test_prots,
                        smile_graph=smile_graph, target_graph=target_graph)

    return test_data
