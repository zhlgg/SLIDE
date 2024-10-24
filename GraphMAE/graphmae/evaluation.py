import copy
from tqdm import tqdm
import torch
import torch.nn as nn

from graphmae.utils import create_optimizer, accuracy


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from reweighting import weight_learner

import numpy as np

class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=False):
        x = z[edge[0]] * z[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


import dgl
def link_prediction_evaluation(model, graph, x, splits, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False, num_layers=2, dataset_name='cora', thin_or_not=False):
    train_data = splits['train'].to(device)
    val_data = splits['valid'].to(device)
    test_data = splits['test'].to(device)

    train_graph = dgl.graph((train_data.edge_index[0], train_data.edge_index[1]), num_nodes=graph.number_of_nodes())
    train_graph = dgl.add_self_loop(train_graph)
    val_graph = dgl.graph((val_data.edge_index[0], val_data.edge_index[1]), num_nodes=graph.number_of_nodes())
    val_graph = dgl.add_self_loop(val_graph)
    test_graph = dgl.graph((test_data.edge_index[0], test_data.edge_index[1]), num_nodes=graph.number_of_nodes())
    test_graph = dgl.add_self_loop(test_graph)
    train_embedding = model.embed(train_graph.to(device), x.to(device))
    val_embedding = model.embed(val_graph.to(device), x.to(device))
    test_embedding = model.embed(test_graph.to(device), x.to(device))

    if num_layers >= 2 and thin_or_not and linear_prob:
        train_embedding = train_embedding[:, ::2]
        val_embedding = val_embedding[:, ::2]
        test_embedding = test_embedding[:, ::2]
    
    model.eval()

    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        if num_layers >= 2 and not thin_or_not:
            encoder = EdgeDecoder(in_feat, 512, num_layers=2, dropout=0.0).to(device)
        elif num_layers >= 2 and thin_or_not:
            encoder = EdgeDecoder(in_feat//2, 512, num_layers=2, dropout=0.0).to(device)
        elif dataset_name != 'pubmed':
            encoder = EdgeDecoder(in_feat, 512, num_layers=2, dropout=0.0).to(device)
        else:
            dropout_rate = 0.2
            encoder = EdgeDecoder(in_feat, 512, num_layers=2, dropout=dropout_rate).to(device)
    else:
        with torch.no_grad():
            x_tmp = model.embed(graph.to(device), x.to(device))
            in_feat = x_tmp.shape[1]
        encoder = model.encoder
        if num_layers >= 2 and not thin_or_not:
            encoder2 = EdgeDecoder(in_feat, 512, num_layers=2, dropout=0.0).to(device)
        elif num_layers >= 2 and thin_or_not:
            encoder2 = EdgeDecoder(in_feat, 512, num_layers=2, dropout=0.0).to(device)
        elif dataset_name != 'pubmed':
            encoder2 = EdgeDecoder(in_feat, 512, num_layers=2, dropout=0.0).to(device)
        else:
            dropout_rate = 0.2
            encoder2 = EdgeDecoder(in_feat, 512, num_layers=2, dropout=dropout_rate).to(device)

    if linear_prob:
        num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    else:
        num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad] + [p.numel() for p in encoder2.parameters() if  p.requires_grad]
    print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    num_model_params = [p.numel() for p in model.parameters()]
    print(f"num parameters for model: {sum(num_model_params)}")

    encoder.to(device)
    if not linear_prob:
        encoder2.to(device)
    if linear_prob:
        optimizer_f = torch.optim.Adam(encoder.parameters(), lr=lr_f, weight_decay=weight_decay_f)
    else:
        # wzb new edit
        if dataset_name == 'cora' or dataset_name == 'citeseer':
            optimizer_f_encoder = torch.optim.Adam(encoder.parameters(), lr=lr_f, weight_decay=0)
            optimizer_f_encoder2 = torch.optim.Adam(encoder2.parameters(), lr=0.01, weight_decay=0.000005)
        else:
            optimizer_f = torch.optim.Adam(list(encoder.parameters()) + list(encoder2.parameters()), lr=lr_f, weight_decay=0)
    if linear_prob:
        val_auc, test_auc, val_ap, test_ap = linear_probing_for_transductive_link_prediction(encoder, train_data, val_data, test_data, train_embedding, val_embedding, test_embedding, optimizer_f, max_epoch_f, device, mute)
    else:
        # wzb new edit
        if dataset_name == 'cora' or dataset_name == 'citeseer':
            optimizer_fs = [optimizer_f_encoder, optimizer_f_encoder2]
        else:
            optimizer_fs = [optimizer_f]
        val_auc, test_auc, val_ap, test_ap = fine_tuning_for_transductive_link_prediction([encoder, encoder2], train_graph, val_graph, test_graph, x, train_data, val_data, test_data, optimizer_fs, max_epoch_f, device, mute)
    return val_auc, test_auc, val_ap, test_ap

def get_link_labels(pos_edge_label_index, neg_edge_index, device):
    num_links = pos_edge_label_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
    link_labels[:pos_edge_label_index.size(1)] = 1.
    return link_labels

# wzb new edit
def fine_tuning_for_transductive_link_prediction(models, train_graph, val_graph, test_graph, x, train_data, val_data, test_data, optimizers, max_epoch, device, mute=False):
    if len(optimizers) == 1:
        optimizer = optimizers[0]
    else:
        optimizer, optimizer2 = optimizers
    encoder, encoder2 = models
    best_auc = 0
    best_ap = 0
    best_test_auc = 0
    best_test_ap = 0
    for epoch in range(max_epoch):
        encoder.train()
        encoder2.train()
        neg_edge_index = negative_sampling(
            edge_index=train_data.pos_edge_label_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.pos_edge_label_index.size(1))
        edge_index = torch.cat([train_data.pos_edge_label_index, neg_edge_index], dim=-1)
        train_embedding = encoder(train_graph.to(device), x.to(device))
        link_logits = encoder2(train_embedding, edge_index)
        link_labels = get_link_labels(train_data.pos_edge_label_index, neg_edge_index, device).to(device)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels.view_as(link_logits))
        optimizer.zero_grad()

        # wzb new edit
        if len(optimizers) == 2:
            optimizer2.zero_grad()

        loss.backward(retain_graph=True)
        
        # wzb new edit
        if len(optimizers) == 2:
            optimizer2.step()
        
        optimizer.step()

        encoder.eval()
        encoder2.eval()
        val_edge_index = torch.cat([val_data.pos_edge_label_index, val_data.neg_edge_label_index], dim=-1)
        test_edge_index = torch.cat([test_data.pos_edge_label_index, test_data.neg_edge_label_index], dim=-1)

        val_embedding = encoder(val_graph.to(device), x.to(device))
        val_link_logits = encoder2(val_embedding, val_edge_index)
        test_embedding = encoder(test_graph.to(device), x.to(device))
        test_link_logits = encoder2(test_embedding, test_edge_index)
        val_link_labels = get_link_labels(val_data.pos_edge_label_index, val_data.neg_edge_label_index, device).to(device)
        test_link_labels = get_link_labels(test_data.pos_edge_label_index, test_data.neg_edge_label_index, device).to(device)

        val_auc = roc_auc_score(val_link_labels.cpu().detach().numpy(), val_link_logits.cpu().detach().numpy())
        test_auc = roc_auc_score(test_link_labels.cpu().detach().numpy(), test_link_logits.cpu().detach().numpy())
        val_ap = average_precision_score(val_link_labels.cpu().detach().numpy(), val_link_logits.cpu().detach().numpy())
        test_ap = average_precision_score(test_link_labels.cpu().detach().numpy(), test_link_logits.cpu().detach().numpy())
    
        if val_auc > best_auc:
            best_auc = val_auc
            best_ap = val_ap
            best_test_auc = test_auc
            best_test_ap = test_ap

    if mute:
        print(f"# IGNORE: --- TestAUC: {best_test_auc:.4f}, TestAP: {best_test_ap:.4f} ---")
    else:
        print(f"--- TestAUC: {best_test_auc:.4f}, TestAP: {best_test_ap:.4f} ---")

    return best_auc, best_test_auc, best_ap, best_test_ap


def linear_probing_for_transductive_link_prediction(model, train_data, val_data, test_data, train_embedding, val_embedding, test_embedding, optimizer, max_epoch, device, mute=False):
    best_auc = 0
    best_ap = 0
    best_test_auc = 0
    best_test_ap = 0
    for epoch in range(max_epoch):
        model.train()
        neg_edge_index = negative_sampling(
            edge_index=train_data.pos_edge_label_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.pos_edge_label_index.size(1))
        edge_index = torch.cat([train_data.pos_edge_label_index, neg_edge_index], dim=-1)  # [2,E]
        
        
        link_logits = model(train_embedding, edge_index)
        link_labels = get_link_labels(train_data.pos_edge_label_index, neg_edge_index, device).to(device) 
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels.view_as(link_logits))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    
        model.eval()
        val_edge_index = torch.cat([val_data.pos_edge_label_index, val_data.neg_edge_label_index], dim=-1)
        test_edge_index = torch.cat([test_data.pos_edge_label_index, test_data.neg_edge_label_index], dim=-1)

        val_link_logits = model(val_embedding, val_edge_index)
        test_link_logits = model(test_embedding, test_edge_index)
        val_link_labels = get_link_labels(val_data.pos_edge_label_index, val_data.neg_edge_label_index, device).to(device)
        test_link_labels = get_link_labels(test_data.pos_edge_label_index, test_data.neg_edge_label_index, device).to(device)

        val_auc = roc_auc_score(val_link_labels.cpu().detach().numpy(), val_link_logits.cpu().detach().numpy())
        test_auc = roc_auc_score(test_link_labels.cpu().detach().numpy(), test_link_logits.cpu().detach().numpy())
        val_ap = average_precision_score(val_link_labels.cpu().detach().numpy(), val_link_logits.cpu().detach().numpy())
        test_ap = average_precision_score(test_link_labels.cpu().detach().numpy(), test_link_logits.cpu().detach().numpy())

        if val_auc > best_auc:
            best_auc = val_auc
            best_ap = val_ap
            best_test_auc = test_auc
            best_test_ap = test_ap

    if mute:
        print(f"# IGNORE: --- TestAUC: {best_test_auc:.4f}, TestAP: {best_test_ap:.4f} ---")
    else:
        print(f"--- TestAUC: {best_test_auc:.4f}, TestAP: {best_test_ap:.4f} ---")

    return best_auc, best_test_auc, best_ap, best_test_ap


from sklearn.metrics import f1_score
def node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, epoch_base=1, mute=False, reweight=False, dataset_name=None):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            x, hidden_list = model.embed(graph.to(device), x.to(device), return_hidden=True)
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)
        hidden_list = None

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    # wzb edit
    lr_wd_dict = {
        'cora': {
            'optimizer_f_head': {
                'lr': 0.05,
                'weight_decay': 1e-4,
            },
            'optimizer_f_others': {
                'lr': 1e-7,
                'weight_decay': 0,
            }
        },
        'citeseer': {
            'optimizer_f_head': {
                'lr': 0.02,
                'weight_decay': 0.1,
            },
            'optimizer_f_others': {
                'lr': 1e-6,
                'weight_decay': 1e-3,
            }
        },
        'pubmed': {
            'optimizer_f_head': {
                'lr': 0.05,
                'weight_decay': 0.0,
            },
            'optimizer_f_others': {
                'lr': 1e-6,
                'weight_decay': 0,
            }
        },
        'photo': {
            'optimizer_f_head': {
                'lr': 0.001,
                'weight_decay': 0.05,
            },
            'optimizer_f_others': {
                'lr': 5e-6,
                'weight_decay': 0,
            }
        },
        'computer': {
            'optimizer_f_head': {
                'lr': 0.001,
                'weight_decay': 0.01,
            },
            'optimizer_f_others': {
                'lr': 5e-5,
                'weight_decay': 0,
            }
        },
        'ogbn-arxiv': {
            'optimizer_f_head': {
                'lr': 0.02,
                'weight_decay': 0.0,
            },
            'optimizer_f_others': {
                'lr': 5e-4,
                'weight_decay': 1e-3,
            }
        }
    }
    if not linear_prob:
        optimizer_f_head = torch.optim.Adam(encoder.head.parameters(), lr=lr_wd_dict[dataset_name]['optimizer_f_head']['lr'], weight_decay=lr_wd_dict[dataset_name]['optimizer_f_head']['weight_decay'])  # lr_f weight_decay_f
        optimizer_f_others = torch.optim.Adam([p for p in encoder.parameters()][:-2], lr=lr_wd_dict[dataset_name]['optimizer_f_others']['lr'], weight_decay=lr_wd_dict[dataset_name]['optimizer_f_others']['weight_decay'])  # 0.000001    0.0
        final_acc, estp_acc, final_f1ma, estp_f1ma = linear_probing_for_transductive_node_classiifcation(encoder, graph, x, [optimizer_f_head, optimizer_f_others], max_epoch_f * epoch_base, device, mute, reweight=reweight, dataset_name=dataset_name)
    else:
        optimizer_f_head = torch.optim.Adam(encoder.parameters(), lr=lr_f, weight_decay=weight_decay_f)
        final_acc, estp_acc, final_f1ma, estp_f1ma = linear_probing_for_transductive_node_classiifcation(encoder, graph, x, [optimizer_f_head], max_epoch_f * epoch_base, device, mute, reweight=reweight)
    
    return (final_acc, estp_acc, final_f1ma, estp_f1ma), hidden_list


# wzb edit
from torch.autograd import Variable
from torch.utils.data import DataLoader

def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizers, max_epoch, device, mute=False, reweight=False, dataset_name=None):
    if reweight:
        criterion = torch.nn.CrossEntropyLoss(reduce=False)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if len(optimizers) == 1:
        optimizer_head = optimizers[0]
        optimizer_others = None
    else:
        optimizer_head, optimizer_others = optimizers

        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_others, lr_lambda=scheduler)

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    if reweight:
        batch_size_dict = {
            'cora': 16,
            'citeseer': 120,
            'pubmed': 60,
            'photo': 153,
            'computer': 275,
            'ogbn-arxiv': 90941, 
        }
        train_loader = DataLoader(train_mask.nonzero().squeeze(), batch_size=batch_size_dict[dataset_name], shuffle=True)

    best_val_acc = 0
    best_val_epoch = 0
    best_val_f1ma = 0
    best_test_acc = 0
    best_test_f1ma = 0
    best_model = None

    epoch_iter = range(max_epoch)

    cur_epoch_num = 0
    with tqdm(total=max_epoch, desc='(LR)',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
        for epoch in epoch_iter:
            model.train()
            
            if reweight:
                for nodes in train_loader:
                    optimizer_head.zero_grad()
                    optimizer_others.zero_grad()
                    out, embedding = model(graph, x, keep_embedding=True)
                    pre_features = model.pre_features
                    pre_weight1 = model.pre_weight1
                    if cur_epoch_num == 0:
                        weight1 = Variable(torch.ones(embedding[nodes].size()[0], 1).cuda())
                    else:
                        weight1, pre_features, pre_weight1 = weight_learner(embedding[nodes], pre_features, pre_weight1, epoch, 0, embedding[nodes].size()[0])
                    model.pre_features.data.copy_(pre_features)
                    model.pre_weight1.data.copy_(pre_weight1)
                    
                    loss = criterion(out[nodes], labels[nodes]).view(1, -1).mm(weight1).view(1) / weight1.sum()

                    loss.backward(retain_graph=True)
                    optimizer_head.step()
                    optimizer_others.step()
            else:
                out = model(graph, x)
                loss = criterion(out[train_mask], labels[train_mask])
                
                optimizer_head.zero_grad()
                if optimizer_others is not None:
                    optimizer_others.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_head.step()
                if optimizer_others is not None:
                    optimizer_others.step()

            with torch.no_grad():
                model.eval()
                pred = model(graph, x)
                val_acc = accuracy(pred[val_mask], labels[val_mask])
                val_loss = criterion(pred[val_mask], labels[val_mask])
                test_acc = accuracy(pred[test_mask], labels[test_mask])
                test_loss = criterion(pred[test_mask], labels[test_mask])

                val_f1ma = f1_score(labels[val_mask].cpu().numpy(), pred[val_mask].argmax(dim=1).cpu().numpy(), average='macro')
                test_f1ma = f1_score(labels[test_mask].cpu().numpy(), pred[test_mask].argmax(dim=1).cpu().numpy(), average='macro')
            
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                best_val_f1ma = val_f1ma
                best_test_acc = test_acc
                best_test_f1ma = test_f1ma

                # zhushi
                best_model = copy.deepcopy(model)

            cur_epoch_num += 1

            pbar.set_postfix({'best acc': best_test_acc, 'f1ma': best_test_f1ma, 'loss': loss.item()})
            pbar.update(1)

    
    # zhushi
    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
        estp_test_f1ma = f1_score(labels[test_mask].cpu().numpy(), pred[test_mask].argmax(dim=1).cpu().numpy(), average='macro')

        class_num = {}
        total_class_number = labels.max().item() + 1
        for i in range(total_class_number):
            class_num[i] = 0
        for i in range(labels.size(0)):
            class_num[labels[i].item()] += 1
        class_acc = {}
        class_f1 = {}
        for i in range(total_class_number):
            class_acc[i] = 0
            class_f1[i] = 0
        for i in range(labels.size(0)):
            if pred[i].argmax().item() == labels[i].item():
                class_acc[labels[i].item()] += 1
        for i in range(total_class_number):
            class_acc[i] /= class_num[i]
        for i in range(total_class_number):
            class_f1[i] = 2 * class_acc[i] * class_num[i] / (class_acc[i] + class_num[i])

    estp_test_acc = best_test_acc
    estp_test_f1ma = best_test_f1ma
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f}, TestF1Ma: {test_f1ma:.4f} , early-stopping-TestF1Ma: {estp_test_f1ma:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f}, TestF1Ma: {test_f1ma:.4f} , early-stopping-TestF1Ma: {estp_test_f1ma:.4f} in epoch {best_val_epoch} --- ")

    return test_acc, estp_test_acc, test_f1ma, estp_test_f1ma


def linear_probing_for_inductive_node_classiifcation(model, x, labels, mask, optimizer, max_epoch, device, mute=False):
    if len(labels.shape) > 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    train_mask, val_mask, test_mask = mask

    best_val_acc = 0
    best_val_epoch = 0
    # wzb edit
    best_val_f1ma = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)  

        best_val_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(None, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(None, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            val_f1ma = f1_score(labels[val_mask].cpu().numpy(), pred[val_mask].argmax(dim=1).cpu().numpy(), average='macro')
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
            test_f1ma = f1_score(labels[test_mask].cpu().numpy(), pred[test_mask].argmax(dim=1).cpu().numpy(), average='macro')
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_f1ma = val_f1ma
            best_model = copy.deepcopy(model)

    best_model.eval()
    with torch.no_grad():
        pred = best_model(None, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
        estp_test_f1ma = f1_score(labels[test_mask].cpu().numpy(), pred[test_mask].argmax(dim=1).cpu().numpy(), average='macro')
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, TestF1ma: {test_f1ma:.4f}, early-stopping-TestF1ma: {estp_test_f1ma:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, TestF1ma: {test_f1ma:.4f}, early-stopping-TestF1ma: {estp_test_f1ma:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}")

    return test_acc, estp_test_acc, test_f1ma, estp_test_f1ma


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits
