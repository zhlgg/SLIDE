import numpy as np
import functools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool_)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
def label_classification_train(model, dataset_name, data, edge_index, nodeclas_weight_decay, base=1, device='cpu'):
    @torch.no_grad()
    def test(loader):
        clf.eval()
        logits = []
        labels = []
        logits.append(clf(embedding))
        labels.append(y)
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        macro = f1_score(labels, logits, average="macro")

        return {
                'acc': (logits == labels).float().mean().item(),
                'F1Ma': macro,
            }

    batch_size = 128
        
    train_loader = DataLoader(data.train_mask.nonzero().squeeze(), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data.test_mask.nonzero().squeeze(), batch_size=20000)
    val_loader = DataLoader(data.val_mask.nonzero().squeeze(), batch_size=20000)

    data = data.to(device)
    y = data.y.squeeze()
    edge_index = edge_index.to(device)
    model.eval()
    model = model.to(device)

    embedding, hiddens = model(data.x, edge_index, return_hidden=True)

    loss_fn = nn.CrossEntropyLoss()
    clf = nn.Linear(embedding.size(1), y.max().item() + 1).to(device)

    num_finetune_params = [p.numel() for p in clf.parameters() if  p.requires_grad]
    print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    epoch_dict = {
        'Cora': 1000,  # 1000
        'CiteSeer': 1000,  # 1000
        'PubMed': 500,  # 500
        'Photo': 500,  # 500
        'Computers': 500,  # 500
    }

    learning_rate_dict = {
        'Cora': 0.05,  # 0.05
        'CiteSeer': 0.5,  # 0.5
        'PubMed': 0.05,  # 0.05
        'Photo': 0.05,  # 0.05
        'Computers': 0.5,  # 0.5
    }

    print('Start Training (Node Classification)...')
    results = []
    f1mas = []
    
    for run in range(1, 5+1):
        nn.init.xavier_uniform_(clf.weight.data)
        nn.init.zeros_(clf.bias.data)
        optimizer = torch.optim.Adam(clf.parameters(), 
                                     lr=learning_rate_dict[dataset_name], 
                                     weight_decay=nodeclas_weight_decay)

        best_val_metric = test_metric = 0
        test_num = 1
        total_epoch_num = int(epoch_dict[dataset_name] * base)
        with tqdm(total=total_epoch_num, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(1, total_epoch_num + 1):
                clf.train()

                # for nodes in train_loader:
                #     optimizer.zero_grad()
                #     loss_fn(clf(embedding[nodes]), y[nodes]).backward(retain_graph=True)
                #     optimizer.step()
                optimizer.zero_grad()
                loss_fn(clf(embedding[data.train_mask]), y[data.train_mask]).backward(retain_graph=True)
                optimizer.step()
                
                if epoch % test_num == 0:
                    val_result = test(val_loader)
                    val_metric, val_f1ma = val_result['acc'], val_result['F1Ma']
                    test_result = test(test_loader)
                    test_metric, test_f1ma = test_result['acc'], test_result['F1Ma']
                    if val_metric > best_val_metric:
                        best_val_metric = val_metric
                        best_test_metric = test_metric
                        best_f1ma = test_f1ma
                    pbar.set_postfix({'best acc': best_test_metric, 'f1ma': best_f1ma})
                    pbar.update(test_num)

        results.append(best_test_metric)
        f1mas.append(best_f1ma)
        print(f'Runs {run}: accuracy {best_test_metric:.2%}')
        print(f'Runs {run}: F1Ma {best_f1ma:.2%}')
                          
    print(f'Node Classification Results ({5} runs):\n'
          f'Accuracy: {np.mean(results):.4}±{np.std(results):.4}\n'
          f'F1Ma: {np.mean(f1mas):.4}±{np.std(f1mas):.4}')

    return hiddens


# wzb edit

from torch.autograd import Variable
from reweighting import weight_learner

def label_classification_train_fine_tune(model, dataset_name, data, edge_index, nodeclas_weight_decay_model, nodeclas_weight_decay_clf, base=1, device='cpu', reweight=False):
    @torch.no_grad()
    def test(loader, model):
        model.eval()
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            logits.append(clf(model(data.x, data.edge_index))[nodes])
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        macro = f1_score(labels, logits, average="macro")

        return {
                'acc': (logits == labels).float().mean().item(),
                'F1Ma': macro,
            }

    epoch_dict = {
        'Cora': 400,  # 200
        'CiteSeer': 100,  # 100
        'PubMed': 200,  # 200
        'Photo': 250,  # 250
        'Computers': 200,  # 200
    }
    
    batch_size_dict = {
        'Cora': 16,  # 16
        'CiteSeer': 8,  # 8
        'PubMed': 60,  # 60
        'Photo': 153,  # 153
        'Computers': 275,  # 275
    }

    learning_rate_model_dict = {
        'Cora': 1e-7,  # 1e-7
        'CiteSeer': 1e-8,  # 1e-8
        'PubMed': 1e-6,  # 1e-6
        'Photo': 5e-5,  # 5e-5
        'Computers': 5e-4,  # 5e-4
    }

    learning_rate_clf_dict = {
        'Cora': 0.02,  # 0.02
        'CiteSeer': 0.01,  # 0.01
        'PubMed': 0.02,  # 0.02
        'Photo': 0.02,  # 0.02
        'Computers': 0.1,  # 0.1
    }

    batch_size = batch_size_dict[dataset_name]
        
    train_loader = DataLoader(data.train_mask.nonzero().squeeze(), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data.test_mask.nonzero().squeeze(), batch_size=20000)
    val_loader = DataLoader(data.val_mask.nonzero().squeeze(), batch_size=20000)

    data = data.to(device)
    y = data.y.squeeze()
    edge_index = edge_index.to(device)
    model.eval()
    model = model.to(device)

    if reweight:
        loss_fn = nn.CrossEntropyLoss(reduce=False)
    else:
        loss_fn = nn.CrossEntropyLoss()
    clf = nn.Linear(model.out_channels, y.max().item() + 1).to(device)

    num_finetune_params = [p.numel() for p in clf.parameters() if  p.requires_grad] + [p.numel() for p in model.parameters() if  p.requires_grad]
    print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    print('Start Training (Node Classification)...')
    results = []
    f1mas = []

    cur_model_state_dict = model.state_dict()
    torch.save(cur_model_state_dict, f'{dataset_name}.pt')
    runs = 5
    for run in range(1, runs+1):
        nn.init.xavier_uniform_(clf.weight.data)
        nn.init.zeros_(clf.bias.data)

        optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate_model_dict[dataset_name], weight_decay=nodeclas_weight_decay_model)
        optimizer_clf = torch.optim.Adam(clf.parameters(), lr=learning_rate_clf_dict[dataset_name], weight_decay=nodeclas_weight_decay_clf)

        best_val_metric = test_metric = 0
        test_num = 1
        best_test_metric, best_f1ma = 0, 0
        total_epoch_num = int(epoch_dict[dataset_name] * base)
        with tqdm(total=total_epoch_num, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:

            for epoch in range(1, total_epoch_num + 1):
                clf.train()
                model.train()
                
                # optimizer.zero_grad()
                # loss_fn(clf(model(data.x, data.edge_index)[data.train_mask]), y_train).backward()
                # optimizer.step()

                for nodes in train_loader:
                    optimizer_model.zero_grad()
                    optimizer_clf.zero_grad()

                    if reweight:
                        embedding = model(data.x, edge_index)[nodes]
                        output = clf(embedding)

                        pre_features = model.pre_features
                        pre_weight1 = model.pre_weight1

                        # print(epoch)
                        # print(pre_features.shape)
                        # print(pre_weight1.shape)
                        # print(embedding.shape)
                        # print(nodes.shape)

                        if epoch == 1:
                            weight1 = Variable(torch.ones(embedding.size()[0], 1).cuda())
                        else:
                            weight1, pre_features, pre_weight1 = weight_learner(embedding, pre_features, pre_weight1, epoch, 0, embedding.size()[0])

                        model.pre_features.data.copy_(pre_features)
                        model.pre_weight1.data.copy_(pre_weight1)

                        if epoch % 20 == 0:
                            print('weight1', weight1.max(), weight1.min())
                    else:
                        output = clf(model(data.x, edge_index)[nodes])
                    
                    if reweight:
                        # print(output.shape)
                        # print(y[nodes].shape)
                        # print(weight1.shape)
                        loss = loss_fn(output, y[nodes]).view(1, -1).mm(weight1).view(1) / weight1.sum()
                    else:
                        loss = loss_fn(output, y[nodes])
                    
                    loss.backward(retain_graph=True)

                    optimizer_clf.step()
                    optimizer_model.step()

                # for nodes in train_loader:
                #     optimizer_model.zero_grad()
                #     optimizer_clf.zero_grad()
                #     loss_fn(clf(model(data.x, edge_index)[nodes]), y[nodes]).backward()
                #     optimizer_clf.step()
                #     optimizer_model.step()
                    
                if epoch % test_num == 0:
                    val_result = test(val_loader, model)
                    val_metric, val_f1ma = val_result['acc'], val_result['F1Ma']
                    test_result = test(test_loader, model)
                    test_metric, test_f1ma = test_result['acc'], test_result['F1Ma']
                    if val_metric > best_val_metric:
                        best_val_metric = val_metric
                        best_test_metric = test_metric
                        best_f1ma = test_f1ma
                    pbar.set_postfix({'best acc': best_test_metric, 'f1ma': best_f1ma, 'loss': loss.item()})
                    pbar.update(test_num)

        results.append(best_test_metric)
        f1mas.append(best_f1ma)
        print(f'Runs {run}: accuracy {best_test_metric:.2%}')
        print(f'Runs {run}: F1Ma {best_f1ma:.2%}')

        cur_model_state_dict = torch.load(f'{dataset_name}.pt')
        model.load_state_dict(cur_model_state_dict)

        if reweight:
            model.pre_features = torch.zeros(model.node_number, model.out_channels).to(weight1.device)
            model.pre_weight1 = torch.ones(model.node_number, 1).to(weight1.device)
                          
    print(f'Node Classification Results ({5} runs):\n'
          f'Accuracy: {np.mean(results):.4}±{np.std(results):.4}\n'
          f'F1Ma: {np.mean(f1mas):.4}±{np.std(f1mas):.4}')


@repeat(3)
def label_classification(embeddings, y, ratio):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    acc = np.mean(np.all(y_pred == y_test, axis=1))
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {
        'acc': acc,
        'F1Mi': micro,
        'F1Ma': macro
    }
