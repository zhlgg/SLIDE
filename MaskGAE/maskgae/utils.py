import sys
import os
import torch
import random
import numpy as np
from texttable import Texttable
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def get_dataset(root: str, name: str, transform=None):
    if name in {'arxiv', 'products', 'mag'}:
        from ogb.nodeproppred import PygNodePropPredDataset
        print('loading ogb dataset...')
        dataset = PygNodePropPredDataset(root=root, name=f'ogbn-{name}')
        if name in ['mag']:
            rel_data = dataset[0]
            # We are only interested in paper <-> paper relations.
            data = Data(
                    x=rel_data.x_dict['paper'],
                    edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                    y=rel_data.y_dict['paper'])
            data = transform(data)
            split_idx = dataset.get_idx_split()
            data.train_mask = index_to_mask(split_idx['train']['paper'], data.num_nodes)
            data.val_mask = index_to_mask(split_idx['valid']['paper'], data.num_nodes)
            data.test_mask = index_to_mask(split_idx['test']['paper'], data.num_nodes)
        else:
            data = transform(dataset[0])
            split_idx = dataset.get_idx_split()
            data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
            data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
            data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)

    elif name in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, name)
        data = transform(dataset[0])

    elif name == 'Reddit':
        dataset = Reddit(osp.join(root, name))
        data = transform(dataset[0])
    elif name in {'Photo', 'Computers'}:
        dataset = Amazon(root, name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
        # print(data.edge_index, '???')
    elif name in {'CS', 'Physics'}:
        dataset = Coauthor(root, name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
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
    else:
        raise ValueError(name)
    return data
    
def tab_printer(args):
    """Function to print the logs in a nice tabular format.

    Note
    ----
    Package `Texttable` is required.
    Run `pip install Texttable` if was not installed.

    Parameters
    ----------
    args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k, str(args[k])] for k in keys if not k.startswith('__')])
    return t.draw()
