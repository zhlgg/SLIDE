import argparse
import os
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull, Amazon
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv, GATConv

from model import Encoder, Model, drop_feature
from eval import label_classification, label_classification_train, label_classification_train_fine_tune

from typing import Optional

from torch import Tensor

import sys
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from CKA import CKA, CudaCKA

def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    r"""Converts indices to a mask representation.

    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)
    
def test_new(model: Model, data, edge_index, dataset_name, base=1, device='cpu'):
    model.eval()

    wd_dict = {
        'Cora': 1e-5,  # 1e-5
        'CiteSeer': 5e-3,  # 5e-3
        'PubMed': 1e-5,  # 1e-5
        'Photo': 0,  # 0
        'Computers': 0,  # 0
    }

    hidden_list = label_classification_train(model, dataset_name, data, edge_index, wd_dict[dataset_name], base, device)
    return hidden_list

def test_new_fine_tune(model: Model, data, edge_index, dataset_name, base=1, device='cpu', reweight=False):
    wd_model = {
        'Cora': 0.0,  # 0.0
        'CiteSeer': 0.0,  # 0.0
        'PubMed': 0.0,  # 0.0
        'Photo': 0.0,  # 0.0
        'Computers': 0.0,  # 0.0
    }

    wd_clf = {
        'Cora': 0.0,  # 0.0
        'CiteSeer': 0.01,  # 0.01
        'PubMed': 0,  # 0.0
        'Photo': 0,  # 0.0
        'Computers': 0,  # 0.0
    }

    hidden_list = label_classification_train_fine_tune(model, dataset_name, data, edge_index, wd_model[dataset_name], wd_clf[dataset_name], base, device, reweight=reweight)
    return hidden_list

class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x):
        logits = self.linear(x)
        return logits

from tqdm import tqdm
from sklearn.metrics import f1_score
def mutli_graph_linear_evaluation(in_dim, feat, labels, device, base=1):
    model_clf = LogisticRegression(in_dim, num_classes)
    num_finetune_params = [p.numel() for p in model_clf.parameters() if  p.requires_grad]
    print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    model_clf.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_clf.parameters(), lr=0.01, weight_decay=0.0)

    best_val_acc = 0
    best_val_epoch = 0
    best_val_test_acc = 0
    best_val_f1ma = 0
    best_val_test_f1ma = 0

    epoch_iter = tqdm(range(1000 * base))  # 1000

    for epoch in epoch_iter:
        model_clf.train()
        for x, y in zip(feat["train"], labels["train"]):
            out = model_clf(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

        with torch.no_grad():
            model_clf.eval()
            val_out = []
            test_out = []
            for x, y in zip(feat["val"], labels["val"]):
                val_pred = model_clf(x)
                val_out.append(val_pred)
            val_out = torch.cat(val_out, dim=0).cpu().numpy()
            val_label = torch.cat(labels["val"], dim=0).cpu().numpy()
            val_out = np.where(val_out >= 0, 1, 0)

            for x, y in zip(feat["test"], labels["test"]):
                test_pred = model_clf(x)# 
                test_out.append(test_pred)
            test_out = torch.cat(test_out, dim=0).cpu().numpy()
            test_label = torch.cat(labels["test"], dim=0).cpu().numpy()
            test_out = np.where(test_out >= 0, 1, 0)

            val_acc = f1_score(val_label, val_out, average="micro")
            test_acc = f1_score(test_label, test_out, average="micro")
            val_f1ma = f1_score(val_label, val_out, average="macro")
            test_f1ma = f1_score(test_label, test_out, average="macro")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_test_acc = test_acc
            best_val_f1ma = val_f1ma
            best_val_test_f1ma = test_f1ma

        epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc}, test_acc:{test_acc: .4f}")

    print(f"--- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f}, Final-TestAcc: {test_acc:.4f} --- , Early-stopping-TestF1ma: {best_val_test_f1ma:.4f},  Final-TestF1ma: {test_f1ma:.4f}")

    return test_acc, best_val_test_acc, test_f1ma, best_val_test_f1ma


def test_new_inductive(model: Model, all_datasets, all_edge_lists, num_classes, device, linear_prob=True, base=1):
    model.eval()
    x_all = {"train": [], "val": [], "test": []}
    y_all = {"train": [], "val": [], "test": []}

    all_hiddens = []

    with torch.no_grad():
        for key, datasets, edge_lists in zip(["train", "val", "test"], all_datasets, all_edge_lists):
            feats = [data.ndata['feat'].to(device) for data in datasets]
            labels = [data.ndata['label'].to(device) for data in datasets]
            for ind, (feat, edge_index, label) in enumerate(zip(feats, edge_lists, labels)):
                x, hiddens = model(feat, edge_index, return_hidden=True)
                x_all[key].append(x)
                y_all[key].append(label)
                if ind == 0:
                    all_hiddens.append(hiddens)

    in_dim = x_all["train"][0].shape[1]
    
    final_acc_list = []
    estp_acc_list = []
    final_f1ma_list = []
    estp_f1ma_list = []
    runs_num = 1  # 5
    for i in range(runs_num):
        final_acc, estp_acc, final_f1ma, estp_f1ma = mutli_graph_linear_evaluation(in_dim, x_all, y_all, device, base)
        final_acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)
        final_f1ma_list.append(final_f1ma)
        estp_f1ma_list.append(estp_f1ma)
    final_acc_mean, final_acc_std = np.mean(final_acc_list), np.std(final_acc_list)
    estp_acc_mean, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    final_f1ma_mean, final_f1ma_std = np.mean(final_f1ma_list), np.std(final_f1ma_list)
    estp_f1ma_mean, estp_f1ma_std = np.mean(estp_f1ma_list), np.std(estp_f1ma_list)
    print(f"Final Acc: {final_acc_mean:.4f}+-{final_acc_std:.4f}, Estp Acc: {estp_acc_mean:.4f}+-{estp_acc_std:.4f}, Final F1ma: {final_f1ma_mean:.4f}+-{final_f1ma_std:.4f}, Estp F1ma: {estp_f1ma_mean:.4f}+-{estp_f1ma_std:.4f}")
    return all_hiddens


import numpy as np
def set_random_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def uniform_element_selection(wt, s_shape, device):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"  # determine whether teacher is larger than student on this dimension
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim] - 1, s_shape[dim]))
        indices = indices.to(torch.int64).to(device)
        ws = torch.index_select(ws, dim, indices)
    assert ws.shape == s_shape
    return ws

def get_slim_weight(teacher_weights, student_weights, device):
    weight_selection = {}
    for key in student_weights.keys():
        # We don't perform weight selection on classification head by default. Remove this constraint if target dataset is the same as teacher's.
        if "head" in key:
            continue
        # First-N layer selection is implicitly applied here
        weight_selection[key] = uniform_element_selection(teacher_weights[key], student_weights[key].shape, device)
    return weight_selection

def get_cka_score_between_models(hidden_list1, hidden_list2, cka):
    cka_score_array = []
    for i in range(min(len(hidden_list1), len(hidden_list2))):
        cka_score_array.append([])
        for j in range(min(len(hidden_list1[i]), len(hidden_list2[i]))):
            cka_score = cka.linear_CKA(hidden_list1[i][j].cpu().detach().numpy(), hidden_list2[i][j].cpu().detach().numpy())
            cka_score_array[i].append(cka_score)
    return cka_score_array

def get_cka_score_between_layers(hidden_list, data_feats, cka):
    cka_score_array = []
    for ind, (hidden, data_feat) in enumerate(zip(hidden_list, data_feats)):
        cka_score_array.append([cka.linear_CKA(data_feat.cpu().detach().numpy(), hidden[0].cpu().detach().numpy())])
        for i in range(len(hidden) - 1):
            cka_score = cka.linear_CKA(hidden[i].cpu().detach().numpy(), hidden[i + 1].cpu().detach().numpy())
            cka_score_array[ind].append(cka_score)
    return cka_score_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    set_random_seed(config['seed'])

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv, 'GATConv': GATConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    def get_dataset(path, name):
        name = 'dblp' if name == 'DBLP' else name

        if name == 'Cora' or name == 'CiteSeer' or name == 'PubMed':
            dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
            return dataset[0], dataset
        elif name == 'Computers' or name == 'Photo':
            dataset =  Amazon(path, name, transform=T.NormalizeFeatures())

            data = dataset[0]
            data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)

            return data, dataset
        elif name == 'ogbn-arxiv':
            from ogb.nodeproppred import PygNodePropPredDataset
            dataset = PygNodePropPredDataset(root=path, name=name)
            data = dataset[0]
            split_idx = dataset.get_idx_split()
            data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
            data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
            data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)
            return data, dataset
        elif name == 'ppi':
            from dgl.data.ppi import PPIDataset
            from dgl.dataloading import GraphDataLoader

            batch_size = 2
            train_datasets = PPIDataset(mode='train')  # 20
            valid_datasets = PPIDataset(mode='valid')  # 2
            test_datasets = PPIDataset(mode='test')  # 2

            train_edge_lists = []
            for g in train_datasets:
                cur_source, cur_target = g.edges()
                cur_edge_index = torch.stack([cur_source, cur_target], dim=0)
                train_edge_lists.append(cur_edge_index)
            
            valid_edge_lists = []
            for g in valid_datasets:
                cur_source, cur_target = g.edges()
                cur_edge_index = torch.stack([cur_source, cur_target], dim=0)
                valid_edge_lists.append(cur_edge_index)

            test_edge_lists = []
            for g in test_datasets:
                cur_source, cur_target = g.edges()
                cur_edge_index = torch.stack([cur_source, cur_target], dim=0)
                test_edge_lists.append(cur_edge_index)

            train_dataloader = GraphDataLoader(train_datasets, batch_size=batch_size)
            valid_dataloader = GraphDataLoader(valid_datasets, batch_size=batch_size, shuffle=False)
            test_dataloader = GraphDataLoader(test_datasets, batch_size=batch_size, shuffle=False)
            eval_train_dataloader = GraphDataLoader(train_datasets, batch_size=batch_size, shuffle=False)
            g = train_datasets[0]
            num_classes = train_datasets.num_labels
            num_features = g.ndata['feat'].shape[1]

            # return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes
            return (train_datasets, train_edge_lists), (valid_datasets, valid_edge_lists), (test_datasets, test_edge_lists), (train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes)

    path = osp.join('.', 'datasets', args.dataset)
    if args.dataset == 'ppi':
        # train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes = get_dataset(path, args.dataset)
        (train_datasets, train_edge_lists), (valid_datasets, valid_edge_lists), (test_datasets, test_edge_lists), (train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes) = get_dataset(path, args.dataset)
    else:
        data, dataset = get_dataset(path, args.dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset == 'ppi':
        for i in range(len(train_edge_lists)):
            train_edge_lists[i] = train_edge_lists[i].to(device)
        for i in range(len(valid_edge_lists)):
            valid_edge_lists[i] = valid_edge_lists[i].to(device)
        for i in range(len(test_edge_lists)):
            test_edge_lists[i] = test_edge_lists[i].to(device)
    else:
        data = data.to(device)

    if args.dataset != 'ppi':
        num_features = dataset.num_features

    # node_number = data.train_mask.nonzero().squeeze().shape[0]
    node_number_dict = {
        'Cora': 24,
        'CiteSeer': 16,
        'PubMed': 60,
        'Photo': 153,
        'Computers': 275,
    }
    node_number = node_number_dict[args.dataset]

    encoder = Encoder(num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers, node_number=node_number).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)

    test_redundancy = True
    test_fine_tune = False
    new_thin_flag = True

    if test_redundancy:
        encoder_half = Encoder(num_features, num_hidden, activation,
                            base_model=base_model, k=num_layers, if_thin=True, thin_base=2, pre_k = config['num_layers'], node_number=node_number).to(device)
        model_half = Model(encoder_half, num_hidden, num_proj_hidden, tau).to(device)

        encoder_quarter = Encoder(num_features, num_hidden, activation,
                                    base_model=base_model, k=num_layers, if_thin=True, thin_base=4, pre_k = config['num_layers'], node_number=node_number).to(device)
        model_quarter = Model(encoder_quarter, num_hidden, num_proj_hidden, tau).to(device)
    
    if new_thin_flag:
        if not test_fine_tune:
            encoder_new_thin = Encoder(num_features, num_hidden, activation,
                                base_model=base_model, k=2, thin_base=1, new_thin=True, pre_k = config['num_layers'], node_number=node_number).to(device)
            model_new_thin = Model(encoder_new_thin, num_hidden, num_proj_hidden, tau).to(device)
        encoder_half_new_thin = Encoder(num_features, num_hidden, activation,
                            base_model=base_model, k=2, if_thin=True, thin_base=2, new_thin=True, pre_k = config['num_layers'], node_number=node_number).to(device)
        model_half_new_thin = Model(encoder_half_new_thin, num_hidden, num_proj_hidden, tau).to(device)
        if not test_fine_tune:
            encoder_quarter_new_thin = Encoder(num_features, num_hidden, activation,
                                        base_model=base_model, k=2, if_thin=True, thin_base=4, new_thin=True, pre_k = config['num_layers'], node_number=node_number).to(device)
            model_quarter_new_thin = Model(encoder_quarter_new_thin, num_hidden, num_proj_hidden, tau).to(device)

    print('encoder paramters:', sum(p.numel() for p in encoder.parameters()))
    print('model paramters:', sum(p.numel() for p in model.parameters()))
    if test_redundancy:
        print('encoder thin paramters:', sum(p.numel() for p in encoder_half.parameters()))
        print('model thin paramters:', sum(p.numel() for p in model_half.parameters()))
        print('encoder quarter paramters:', sum(p.numel() for p in encoder_quarter.parameters()))
        print('model quarter paramters:', sum(p.numel() for p in model_quarter.parameters()))
    
    if new_thin_flag:
        if not test_fine_tune:
            print('encoder new thin paramters:', sum(p.numel() for p in encoder_new_thin.parameters()))
            print('model new thin paramters:', sum(p.numel() for p in model_new_thin.parameters()))
        print('encoder half new thin paramters:', sum(p.numel() for p in encoder_half_new_thin.parameters()))
        print('model half new thin paramters:', sum(p.numel() for p in model_half_new_thin.parameters()))
        if not test_fine_tune:
            print('encoder quarter new thin paramters:', sum(p.numel() for p in encoder_quarter_new_thin.parameters()))
            print('model quarter new thin paramters:', sum(p.numel() for p in model_quarter_new_thin.parameters()))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if args.dataset == 'ppi':
        start = t()
        prev = start
        data_list = [data.ndata['feat'].to(device) for data in train_datasets]
        for epoch in range(1, num_epochs + 1):
            total_loss = 0
            for data, edge_index in zip(data_list, train_edge_lists):
                loss = train(model, data, edge_index)
                total_loss += loss
            total_loss /= len(train_datasets)
            now = t()
            print(f'(T) | Epoch={epoch:03d}, loss={total_loss:.4f}, '
                f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            prev = now
    else:
        start = t()
        prev = start
        for epoch in range(1, num_epochs + 1):
            loss = train(model, data.x, data.edge_index)

            now = t()
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            prev = now

    print("=== Final ===")

    test_old = False

    if args.dataset != 'ppi':
        print('=== Full ===')
        if test_old:
            test(model, data.x, data.edge_index, data.y, final=True)
        else:
            hidden_list = test_new(model, data, data.edge_index, args.dataset, 1, device)  # 1

        if test_redundancy:
            print("=== Half ===")
            weight_selection = get_slim_weight(model.state_dict(), model_half.state_dict(), device)
            model_half.load_state_dict(weight_selection)
            if test_old:
                test(model_half, data.x, data.edge_index, data.y, final=True)
            else:
                thin_hidden_list = test_new(model_half, data, data.edge_index, args.dataset, 1, device)  # 2

            print("=== Quarter ===")
            weight_selection = get_slim_weight(model.state_dict(), model_quarter.state_dict(), device)
            model_quarter.load_state_dict(weight_selection)
            if test_old:
                test(model_quarter, data.x, data.edge_index, data.y, final=True)
            else:
                thinner_hidden_list = test_new(model_quarter, data, data.edge_index, args.dataset, 1, device)

        if new_thin_flag:
            if not test_fine_tune:
                print("=== New Thin ===")
                weight_selection = get_slim_weight(model.state_dict(), model_new_thin.state_dict(), device)
                model_new_thin.load_state_dict(weight_selection)
                if test_old:
                    test(model_new_thin, data.x, data.edge_index, data.y, final=True)
                else:
                    thin_new_hidden_list = test_new(model_new_thin, data, data.edge_index, args.dataset, 1, device)  # 1

            print("=== Half New Thin ===")
            weight_selection = get_slim_weight(model.state_dict(), model_half_new_thin.state_dict(), device)
            model_half_new_thin.load_state_dict(weight_selection)
            if test_old:
                test(model_half_new_thin, data.x, data.edge_index, data.y, final=True)
            else:
                thin_new_thin_hidden_list = test_new(model_half_new_thin, data, data.edge_index, args.dataset, 1, device)  # 2

            if not test_fine_tune:
                print("=== Quarter New Thin ===")
                weight_selection = get_slim_weight(model.state_dict(), model_quarter_new_thin.state_dict(), device)
                model_quarter_new_thin.load_state_dict(weight_selection)
                if test_old:
                    test(model_quarter_new_thin, data.x, data.edge_index, data.y, final=True)
                else:
                    thinner_new_thin_hidden_list = test_new(model_quarter_new_thin, data, data.edge_index, args.dataset, 1, device)

        if not test_fine_tune:
            cka = CKA()
            cka_score_array_between_models = {}
            cka_score_array_between_layers = {}

            cka_max_node_num = 10000
            x = data.x

            if test_redundancy:
                for cur_layer in range(len(hidden_list)):
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].cpu().detach().numpy(), hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model'][0].append(cka_score_between_data_and_layer0)
                    if cur_layer != len(hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy(), hidden_list[cur_layer + 1][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model']) < cur_layer + 2:
                                cka_score_array_between_layers['model'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model'][cur_layer + 1].append(cka_score_between_layer)
                for cur_layer in range(len(hidden_list)):
                    cka_score_between_model_and_thin = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy(), thin_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy())
                    if 'model_thin' not in cka_score_array_between_models:
                        cka_score_array_between_models['model_thin'] = [[cka_score_between_model_and_thin]]
                    else:
                        if len(cka_score_array_between_models['model_thin']) < cur_layer + 1:
                            cka_score_array_between_models['model_thin'].append([cka_score_between_model_and_thin])
                        else:
                            cka_score_array_between_models['model_thin'][cur_layer].append(cka_score_between_model_and_thin)
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].cpu().detach().numpy(), thin_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model_thin' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_thin'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model_thin'][0].append(cka_score_between_data_and_layer0)
                    
                    if cur_layer != len(hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(thin_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy(), thin_hidden_list[cur_layer + 1][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model_thin' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_thin'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model_thin']) < cur_layer + 2:
                                cka_score_array_between_layers['model_thin'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model_thin'][cur_layer + 1].append(cka_score_between_layer)

                for cur_layer in range(len(hidden_list)):
                    cka_score_between_model_and_thinner = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy(), thinner_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy())
                    if 'model_thinner' not in cka_score_array_between_models:
                        cka_score_array_between_models['model_thinner'] = [[cka_score_between_model_and_thinner]]
                    else:
                        if len(cka_score_array_between_models['model_thinner']) < cur_layer + 1:
                            cka_score_array_between_models['model_thinner'].append([cka_score_between_model_and_thinner])
                        else:
                            cka_score_array_between_models['model_thinner'][cur_layer].append(cka_score_between_model_and_thinner)
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].cpu().detach().numpy(), thinner_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model_thinner' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_thinner'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model_thinner'][0].append(cka_score_between_data_and_layer0)
                    
                    if cur_layer != len(hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(thinner_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy(), thinner_hidden_list[cur_layer + 1][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model_thinner' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_thinner'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model_thinner']) < cur_layer + 2:
                                cka_score_array_between_layers['model_thinner'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model_thinner'][cur_layer + 1].append(cka_score_between_layer)

            if new_thin_flag:
                for cur_layer in range(len(thin_new_hidden_list)):
                    cka_score_between_model_and_new_thin = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy(), thin_new_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy())
                    if 'model_new' not in cka_score_array_between_models:
                        cka_score_array_between_models['model_new'] = [[cka_score_between_model_and_new_thin]]
                    else:
                        if len(cka_score_array_between_models['model_new']) < cur_layer + 1:
                            cka_score_array_between_models['model_new'].append([cka_score_between_model_and_new_thin])
                        else:
                            cka_score_array_between_models['model_new'][cur_layer].append(cka_score_between_model_and_new_thin)
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].cpu().detach().numpy(), thin_new_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model_new' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model_new'][0].append(cka_score_between_data_and_layer0)
                    
                    if cur_layer != len(thin_new_hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(thin_new_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy(), thin_new_hidden_list[cur_layer + 1][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model_new' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model_new']) < cur_layer + 2:
                                cka_score_array_between_layers['model_new'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model_new'][cur_layer + 1].append(cka_score_between_layer)

                for cur_layer in range(len(thin_new_thin_hidden_list)):
                    cka_score_between_model_and_thin_new_thin = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy(), thin_new_thin_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy())
                    if 'model_new_thin' not in cka_score_array_between_models:
                        cka_score_array_between_models['model_new_thin'] = [[cka_score_between_model_and_thin_new_thin]]
                    else:
                        if len(cka_score_array_between_models['model_new_thin']) < cur_layer + 1:
                            cka_score_array_between_models['model_new_thin'].append([cka_score_between_model_and_thin_new_thin])
                        else:
                            cka_score_array_between_models['model_new_thin'][cur_layer].append(cka_score_between_model_and_thin_new_thin)
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].cpu().detach().numpy(), thin_new_thin_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model_new_thin' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new_thin'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model_new_thin'][0].append(cka_score_between_data_and_layer0)
                    
                    if cur_layer != len(thin_new_thin_hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(thin_new_thin_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy(), thin_new_thin_hidden_list[cur_layer + 1][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model_new_thin' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new_thin'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model_new_thin']) < cur_layer + 2:
                                cka_score_array_between_layers['model_new_thin'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model_new_thin'][cur_layer + 1].append(cka_score_between_layer)

                for cur_layer in range(len(thinner_new_thin_hidden_list)):
                    cka_score_between_model_and_thinner_new_thin = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy(), thinner_new_thin_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy())
                    if 'model_new_thinner' not in cka_score_array_between_models:
                        cka_score_array_between_models['model_new_thinner'] = [[cka_score_between_model_and_thinner_new_thin]]
                    else:
                        if len(cka_score_array_between_models['model_new_thinner']) < cur_layer + 1:
                            cka_score_array_between_models['model_new_thinner'].append([cka_score_between_model_and_thinner_new_thin])
                        else:
                            cka_score_array_between_models['model_new_thinner'][cur_layer].append(cka_score_between_model_and_thinner_new_thin)
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].cpu().detach().numpy(), thinner_new_thin_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model_new_thinner' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new_thinner'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model_new_thinner'][0].append(cka_score_between_data_and_layer0)
                    
                    if cur_layer != len(thinner_new_thin_hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(thinner_new_thin_hidden_list[cur_layer][:cka_max_node_num, :].cpu().detach().numpy(), thinner_new_thin_hidden_list[cur_layer + 1][:cka_max_node_num, :].cpu().detach().numpy())
                        if 'model_new_thinner' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new_thinner'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model_new_thinner']) < cur_layer + 2:
                                cka_score_array_between_layers['model_new_thinner'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model_new_thinner'][cur_layer + 1].append(cka_score_between_layer)
            for i in cka_score_array_between_models:
                for j in range(len(cka_score_array_between_models[i])):
                    print(f'cka_score_between_models_{i}_layer_{j}: {np.mean(cka_score_array_between_models[i][j]):.4f}±{np.std(cka_score_array_between_models[i][j]):.4f}')

            for i in cka_score_array_between_layers:
                for j in range(len(cka_score_array_between_layers[i])):
                    print(f'cka_score_between_models_{i}_layer_{j}_and_layer_{j+1}: {np.mean(cka_score_array_between_layers[i][j]):.4f}±{np.std(cka_score_array_between_layers[i][j]):.4f}')
        else:
            # root_checkpoint_dir = f'./checkpoint/{dataset_name}'
            # if not os.path.exists(root_checkpoint_dir):
            #     os.makedirs(root_checkpoint_dir)
            # model_encoder_save_path = os.path.join(root_checkpoint_dir, f'model_encoder_state_dict_{dataset_name}.pt')
            # model_thin_new_thin_encoder_save_path = os.path.join(root_checkpoint_dir, f'model_thin_new_thin_encoder_state_dict_{dataset_name}.pt')
            # torch.save(model.encoder.state_dict(), model_encoder_save_path)
            # torch.save(model_half_new_thin.encoder.state_dict(), model_thin_new_thin_encoder_save_path)
            
            test_new_fine_tune(model_half_new_thin.encoder, data, data.edge_index, args.dataset, 1, device)
            
            weight_selection = get_slim_weight(model.state_dict(), model_half_new_thin.state_dict(), device)
            model_half_new_thin.load_state_dict(weight_selection)

            reweight_best_fine_tune_base_dict = {
                'Cora': 0.625,
                'CiteSeer': 1,
                'PubMed': 0.6,
                'Computers': 1,
                'Photo': 1,
            }

            full_best_fine_tune_base_dict = {
                'Cora': 0.5,
                'CiteSeer': 1,
                'PubMed': 1,
                'Computers': 1,
                'Photo': 1,
            }

            test_new_fine_tune(model.encoder, data, data.edge_index, args.dataset, full_best_fine_tune_base_dict[args.dataset], device)  # 1
            test_new_fine_tune(model_half_new_thin.encoder, data, data.edge_index, args.dataset, reweight_best_fine_tune_base_dict[args.dataset], device, reweight=True)
            
    else:
        print('=== Full ===')
        hidden_list = test_new_inductive(model, [train_datasets, valid_datasets, test_datasets], [train_edge_lists, valid_edge_lists, test_edge_lists], num_classes, device, base=1)  # 2

        if test_redundancy:
            weight_selection = get_slim_weight(model.state_dict(), model_half.state_dict(), device)
            model_half.load_state_dict(weight_selection)
            thin_hidden_list = test_new_inductive(model_half, [train_datasets, valid_datasets, test_datasets], [train_edge_lists, valid_edge_lists, test_edge_lists], num_classes, device, base=1)  # 4
            
            weight_selection = get_slim_weight(model.state_dict(), model_quarter.state_dict(), device)
            model_quarter.load_state_dict(weight_selection)
            thinner_hidden_list = test_new_inductive(model_quarter, [train_datasets, valid_datasets, test_datasets], [train_edge_lists, valid_edge_lists, test_edge_lists], num_classes, device, base=1)  # 4

        if new_thin_flag:
            weight_selection = get_slim_weight(model.state_dict(), model_new_thin.state_dict(), device)
            model_new_thin.load_state_dict(weight_selection)
            thin_new_hidden_list = test_new_inductive(model_new_thin, [train_datasets, valid_datasets, test_datasets], [train_edge_lists, valid_edge_lists, test_edge_lists], num_classes, device, base=1)  # 4
            
            weight_selection = get_slim_weight(model.state_dict(), model_half_new_thin.state_dict(), device)
            model_half_new_thin.load_state_dict(weight_selection)
            thin_new_thin_hidden_list = test_new_inductive(model_half_new_thin, [train_datasets, valid_datasets, test_datasets], [train_edge_lists, valid_edge_lists, test_edge_lists], num_classes, device, base=1)  # 4
            
            weight_selection = get_slim_weight(model.state_dict(), model_quarter_new_thin.state_dict(), device)
            model_quarter_new_thin.load_state_dict(weight_selection)
            thinner_new_thin_hidden_list = test_new_inductive(model_quarter_new_thin, [train_datasets, valid_datasets, test_datasets], [train_edge_lists, valid_edge_lists, test_edge_lists], num_classes, device, base=1)  # 4

        cka = CKA()

        cka_max_node_num = 10000  # 10000
        train_data_feat = train_datasets[0].ndata['feat']
        valid_data_feat = valid_datasets[0].ndata['feat']
        test_data_feat = test_datasets[0].ndata['feat']

        data_feats = [train_data_feat, valid_data_feat, test_data_feat]
        id2mode = {0: 'train', 1: 'valid', 2: 'test'}

        cka_scores_between_model_and_thin = get_cka_score_between_models(hidden_list, thin_hidden_list, cka)        
        print('-' * 50 + '\n')
        for i in range(len(cka_scores_between_model_and_thin)):
            for j in range(len(cka_scores_between_model_and_thin[i])):
                print(f'cka_score_between_model_and_thin_layer_{j} in {id2mode[i]}: {cka_scores_between_model_and_thin[i][j]}')
        print('-' * 50 + '\n')
        cka_scores_between_model_and_thinner = get_cka_score_between_models(hidden_list, thinner_hidden_list, cka)
        for i in range(len(cka_scores_between_model_and_thinner)):
            for j in range(len(cka_scores_between_model_and_thinner[i])):
                print(f'cka_score_between_model_and_thinner_layer_{j} in {id2mode[i]}: {cka_scores_between_model_and_thinner[i][j]}')
        print('-' * 50 + '\n')
        cka_scores_between_model_and_new_thin = get_cka_score_between_models(hidden_list, thin_new_hidden_list, cka)
        for i in range(len(cka_scores_between_model_and_new_thin)):
            for j in range(len(cka_scores_between_model_and_new_thin[i])):
                print(f'cka_score_between_model_and_new_thin_layer_{j} in {id2mode[i]}: {cka_scores_between_model_and_new_thin[i][j]}')
        print('-' * 50 + '\n')
        cka_scores_between_model_and_thin_new_thin = get_cka_score_between_models(hidden_list, thin_new_thin_hidden_list, cka)
        for i in range(len(cka_scores_between_model_and_thin_new_thin)):
            for j in range(len(cka_scores_between_model_and_thin_new_thin[i])):
                print(f'cka_score_between_model_and_thin_new_thin_layer_{j} in {id2mode[i]}: {cka_scores_between_model_and_thin_new_thin[i][j]}')
        print('-' * 50 + '\n')
        cka_scores_between_model_and_thinner_new_thin = get_cka_score_between_models(hidden_list, thinner_new_thin_hidden_list, cka)
        for i in range(len(cka_scores_between_model_and_thinner_new_thin)):
            for j in range(len(cka_scores_between_model_and_thinner_new_thin[i])):
                print(f'cka_score_between_model_and_thinner_new_thin_layer_{j} in {id2mode[i]}: {cka_scores_between_model_and_thinner_new_thin[i][j]}')
        print('-' * 50 + '\n')
        cka_scores_between_model_layers = get_cka_score_between_layers(hidden_list, data_feats, cka)
        for i in range(len(cka_scores_between_model_layers)):
            for j in range(len(cka_scores_between_model_layers[i])):
                print(f'cka_score_between_model_layer_{j-1}_and _layer{j} in {id2mode[i]}: {cka_scores_between_model_layers[i][j]}')
        print('-' * 50 + '\n')
        cka_scores_between_model_thin_layers = get_cka_score_between_layers(thin_hidden_list, data_feats, cka)
        for i in range(len(cka_scores_between_model_thin_layers)):
            for j in range(len(cka_scores_between_model_thin_layers[i])):
                print(f'cka_score_between_model_thin_layer_{j-1}_and _layer{j} in {id2mode[i]}: {cka_scores_between_model_thin_layers[i][j]}')
        print('-' * 50 + '\n')
        cka_scores_between_model_thinner_layers = get_cka_score_between_layers(thinner_hidden_list, data_feats, cka)
        for i in range(len(cka_scores_between_model_thinner_layers)):
            for j in range(len(cka_scores_between_model_thinner_layers[i])):
                print(f'cka_score_between_model_thinner_layer_{j-1}_and _layer{j} in {id2mode[i]}: {cka_scores_between_model_thinner_layers[i][j]}')
        print('-' * 50 + '\n')
        cka_scores_between_model_new_layers = get_cka_score_between_layers(thin_new_hidden_list, data_feats, cka)
        for i in range(len(cka_scores_between_model_new_layers)):
            for j in range(len(cka_scores_between_model_new_layers[i])):
                print(f'cka_score_between_model_new_layer_{j-1}_and _layer{j} in {id2mode[i]}: {cka_scores_between_model_new_layers[i][j]}')
        print('-' * 50 + '\n')
        cka_scores_between_model_new_thin_layers = get_cka_score_between_layers(thin_new_thin_hidden_list, data_feats, cka)
        for i in range(len(cka_scores_between_model_new_thin_layers)):
            for j in range(len(cka_scores_between_model_new_thin_layers[i])):
                print(f'cka_score_between_model_new_thin_layer_{j-1}_and _layer{j} in {id2mode[i]}: {cka_scores_between_model_new_thin_layers[i][j]}')
        print('-' * 50 + '\n')
        cka_scores_between_model_new_thinner_layers = get_cka_score_between_layers(thinner_new_thin_hidden_list, data_feats, cka)
        for i in range(len(cka_scores_between_model_new_thinner_layers)):
            for j in range(len(cka_scores_between_model_new_thinner_layers[i])):
                print(f'cka_score_between_model_new_thinner_layer_{j-1}_and _layer{j} in {id2mode[i]}: {cka_scores_between_model_new_thinner_layers[i][j]}')
        print('-' * 50 + '\n')
