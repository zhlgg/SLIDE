import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

# custom modules
from maskgae.utils import set_seed, tab_printer, get_dataset
from maskgae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
from maskgae.mask import MaskEdge, MaskPath


# edit reweight
from torch.autograd import Variable
from reweighting import weight_learner

def train_linkpred(model, splits, args, device="cpu"):
    print('Start Training (Link Prediction Pretext Training)...')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_valid = 0
    batch_size = args.batch_size
    
    train_data = splits['train'].to(device)
    valid_data = splits['valid'].to(device)
    test_data = splits['test'].to(device)
    
    model.reset_parameters()
    
    for epoch in tqdm(range(1, 1 + args.epochs)):

        loss = model.train_step(train_data, optimizer,
                                alpha=args.alpha, 
                                batch_size=args.batch_size)
        
        if epoch % args.eval_period == 0:
            valid_auc, valid_ap = model.test_step(valid_data, 
                                                  valid_data.pos_edge_label_index, 
                                                  valid_data.neg_edge_label_index, 
                                                  batch_size=batch_size)
            if valid_auc > best_valid:
                best_valid = valid_auc
                best_epoch = epoch
                torch.save(model.state_dict(), args.save_path)

    model.load_state_dict(torch.load(args.save_path))
    test_auc, test_ap = model.test_step(test_data, 
                                        test_data.pos_edge_label_index, 
                                        test_data.neg_edge_label_index, 
                                        batch_size=batch_size)   
    
    print(f'Link Prediction Pretraining Results:\n'
          f'AUC: {test_auc:.2%}',
          f'AP: {test_ap:.2%}')
    return test_auc, test_ap

from sklearn.metrics import f1_score

def train_nodeclas(model, data, args, base=1, device='cpu'):
    @torch.no_grad()
    def test(loader):
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            logits.append(clf(embedding[nodes]))
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)

        macro = f1_score(labels, logits, average="macro")

        return {
                'acc': (logits == labels).float().mean().item(),
                'F1Ma': macro,
            }

    @torch.no_grad()
    def test_and_get_detail(loader, clf):
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            logits.append(clf(embedding[nodes]))
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)

        macro = f1_score(labels, logits, average="macro")

        class_num = {}
        total_class_number = labels.max().item() + 1
        for i in range(total_class_number):
            class_num[i] = 0
        for i in range(len(labels)):
            class_num[labels[i].item()] += 1

        class_acc = {}
        class_f1 = {}
        for i in range(total_class_number):
            class_acc[i] = 0
            class_f1[i] = 0
        for i in range(len(labels)):
            if labels[i].item() == logits[i].item():
                class_acc[labels[i].item()] += 1
        for i in range(total_class_number):
            class_acc[i] /= class_num[i]
        for i in range(total_class_number):
            class_f1[i] = 2 * class_acc[i] * class_num[i] / (class_acc[i] + class_num[i])

        return {
                'acc': (logits == labels).float().mean().item(),
                'F1Ma': macro,
            }

    if args.dataset in {'arxiv', 'products', 'mag'}:
        batch_size = 4096
    else:
        batch_size = 512
        
    train_loader = DataLoader(data.train_mask.nonzero().squeeze(), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data.test_mask.nonzero().squeeze(), batch_size=20000)
    val_loader = DataLoader(data.val_mask.nonzero().squeeze(), batch_size=20000)

    data = data.to(device)
    y = data.y.squeeze()
    embedding = model.encoder.get_embedding(data.x, data.edge_index)

    if args.l2_normalize:
        embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed    

    loss_fn = nn.CrossEntropyLoss()
    clf = nn.Linear(embedding.size(1), y.max().item() + 1).to(device)

    num_finetune_params = [p.numel() for p in clf.parameters() if  p.requires_grad]
    print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    print('Start Training (Node Classification)...')
    results = []
    f1mas = []
    
    run_num = 5
    for run in range(1, run_num + 1):
        nn.init.xavier_uniform_(clf.weight.data)
        nn.init.zeros_(clf.bias.data)
        optimizer = torch.optim.Adam(clf.parameters(), 
                                     lr=0.01, 
                                     weight_decay=args.nodeclas_weight_decay)

        best_val_metric, test_metric, best_f1ma = 0, 0, 0
        best_clf = None
        
        total_epoch_num = 100 * base

        with tqdm(total=total_epoch_num, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:

            for epoch in range(1, total_epoch_num + 1):
                clf.train()
                for nodes in train_loader:
                    optimizer.zero_grad()
                    loss_fn(clf(embedding[nodes]), y[nodes]).backward()
                    optimizer.step()
                    
                val_result = test(val_loader)
                val_metric, val_f1ma = val_result['acc'], val_result['F1Ma']
                test_result = test(test_loader)
                test_metric, test_f1ma = test_result['acc'], test_result['F1Ma']
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test_metric
                    best_f1ma = test_f1ma
                    best_clf = clf
                pbar.set_postfix({'best acc': best_test_metric, 'f1ma': best_f1ma})
                pbar.update(1)

        results.append(best_test_metric)
        f1mas.append(best_f1ma)

        best_result_detail = test_and_get_detail(test_loader, best_clf)

        print(f'Runs {run}: accuracy {best_test_metric:.2%}')
        print(f'Runs {run}: F1Ma {best_f1ma:.2%}')
                          
    print(f'Node Classification Results ({5} runs):\n'
          f'Accuracy: {np.mean(results):.4}±{np.std(results):.4}\n'
          f'F1Ma: {np.mean(f1mas):.4}±{np.std(f1mas):.4}')

def train_nodeclas_fine_tune(model, data, args, model_learning_rate=1e-4, model_weight_decay=0, clf_learning_rate=0.01, clf_weight_decay=0, epoch_num=100, device='cpu', reweight=False, batch_size_dict=None):
    @torch.no_grad()
    def test(loader, model):
        model.eval()
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            embedding = model.encoder.get_embedding(data.x, data.edge_index)
            if args.l2_normalize:
                embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed 

            logits.append(clf(embedding)[nodes])
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        macro = f1_score(labels, logits, average="macro")

        return {
                'acc': (logits == labels).float().mean().item(),
                'F1Ma': macro,
            }

    @torch.no_grad()
    def test_and_get_detail(loader, model, clf):
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            embedding = model.encoder.get_embedding(data.x, data.edge_index)
            if args.l2_normalize:
                embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed 

            logits.append(clf(embedding[nodes]))
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)

        macro = f1_score(labels, logits, average="macro")

        class_num = {}
        total_class_number = labels.max().item() + 1
        for i in range(total_class_number):
            class_num[i] = 0
        for i in range(len(labels)):
            class_num[labels[i].item()] += 1

        class_acc = {}
        class_f1 = {}
        for i in range(total_class_number):
            class_acc[i] = 0
            class_f1[i] = 0
        for i in range(len(labels)):
            if labels[i].item() == logits[i].item():
                class_acc[labels[i].item()] += 1
        for i in range(total_class_number):
            class_acc[i] /= class_num[i]
        for i in range(total_class_number):
            class_f1[i] = 2 * class_acc[i] * class_num[i] / (class_acc[i] + class_num[i])

        return {
                'acc': (logits == labels).float().mean().item(),
                'F1Ma': macro,
            }

    if args.dataset in {'arxiv', 'products', 'mag'}:
        batch_size = 4096
    else:
        batch_size = 512

    if reweight:
        batch_size = batch_size_dict[args.dataset]
        
    train_loader = DataLoader(data.train_mask.nonzero().squeeze(), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data.test_mask.nonzero().squeeze(), batch_size=20000)
    val_loader = DataLoader(data.val_mask.nonzero().squeeze(), batch_size=20000)

    data = data.to(device)
    y = data.y.squeeze()
    y_train = data.y[data.train_mask].squeeze()

    if reweight:
        loss_fn = nn.CrossEntropyLoss(reduce=False)
    else:
        loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        embedding = model.encoder.get_embedding(data.x, data.edge_index)
        if args.l2_normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
    clf = nn.Linear(embedding.size(1), y.max().item() + 1).to(device)
    model.train()
    clf.train()

    num_finetune_params = [p.numel() for p in clf.parameters() if  p.requires_grad] + [p.numel() for p in model.parameters() if  p.requires_grad]
    print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    print('Start Training (Node Classification)...')
    results = []
    f1mas = []

    cur_model_state_dict = model.state_dict()
    cur_model_save_path = f'{args.dataset}_{args.save_path.split(".")[0]}.pt'
    torch.save(cur_model_state_dict, cur_model_save_path)
    
    run_num = 5
    for run in range(1, run_num+1):
        nn.init.xavier_uniform_(clf.weight.data)
        nn.init.zeros_(clf.bias.data)
        # optimizer should tune model and clf
        optimizer_model = torch.optim.Adam(model.parameters(), lr=model_learning_rate, weight_decay=model_weight_decay)
        optimizer_clf = torch.optim.Adam(clf.parameters(), lr=clf_learning_rate, weight_decay=clf_weight_decay)

        best_val_metric, test_metric, best_f1ma = 0, 0, 0
        best_model, best_clf = None, None

        total_epoch_num = epoch_num

        with tqdm(total=total_epoch_num, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:

            for epoch in range(1, total_epoch_num + 1):
                clf.train()
                model.train()

                for nodes in train_loader:
                    optimizer_model.zero_grad()
                    optimizer_clf.zero_grad()
                    
                    embedding = model.encoder.get_embedding_tune(data.x, data.edge_index)
                    if args.l2_normalize:
                        embedding = F.normalize(embedding, p=2, dim=1)[nodes]
                    else:
                        embedding = embedding[nodes]
                    if reweight:
                        pre_features = model.pre_features
                        pre_weight1 = model.pre_weight1
                        if epoch == 1:
                            weight1 = Variable(torch.ones(embedding.size()[0], 1).cuda())
                        else:
                            weight1, pre_features, pre_weight1 = weight_learner(embedding, pre_features, pre_weight1, epoch, 0, embedding.size()[0])
                        model.pre_features.data.copy_(pre_features)
                        model.pre_weight1.data.copy_(pre_weight1)
                    output = clf(embedding)
                    if reweight:
                        loss = loss_fn(output, y[nodes]).view(1, -1).mm(weight1).view(1) / weight1.sum()
                    else:
                        loss = loss_fn(output, y[nodes])

                    loss.backward(retain_graph=True)
                    
                    
                    optimizer_clf.step()
                    optimizer_model.step()
                    
                val_result = test(val_loader, model)
                val_metric, val_f1ma = val_result['acc'], val_result['F1Ma']
                test_result = test(test_loader, model)
                test_metric, test_f1ma = test_result['acc'], test_result['F1Ma']
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test_metric
                    best_f1ma = test_f1ma
                    best_clf = clf
                    best_model = model
                
                pbar.set_postfix({'best acc': best_test_metric, 'f1ma': best_f1ma, 'loss': loss.item()})
                pbar.update(1)

                if (args.dataset == 'arxiv') and reweight:
                    model.pre_features = torch.zeros(model.node_number, model.encoder.embedding_channels).to(weight1.device)
                    model.pre_weight1 = torch.ones(model.node_number, 1).to(weight1.device)


        results.append(best_test_metric)
        f1mas.append(best_f1ma)

        best_result_detail = test_and_get_detail(test_loader, best_model, best_clf)

        cur_model_state_dict = torch.load(cur_model_save_path)
        model.load_state_dict(cur_model_state_dict)

        if reweight:
            model.pre_features = torch.zeros(model.node_number, model.encoder.embedding_channels).to(weight1.device)
            model.pre_weight1 = torch.ones(model.node_number, 1).to(weight1.device)

        print(f'Runs {run}: accuracy {best_test_metric:.2%}')
        print(f'Runs {run}: F1Ma {best_f1ma:.2%}')
                          
    print(f'Node Classification Results ({5} runs):\n'
          f'Accuracy: {np.mean(results):.4}±{np.std(results):.4}\n'
          f'F1Ma: {np.mean(f1mas):.4}±{np.std(f1mas):.4}')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument("--mask", nargs="?", default="Path", help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')

parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder layers. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=64, help='Channels of hidden representation. (default: 64)')
parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder layers. (default: 128)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument('--alpha', type=float, default=0., help='loss weight for degree prediction. (default: 0.)')

parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size for link prediction training. (default: 2**16)')

parser.add_argument("--start", nargs="?", default="node", help="Which Type to sample starting nodes for random walks, (default: node)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. (default: 500)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=30, help='(default: 30)')
parser.add_argument("--save_path", nargs="?", default="MaskGAE-NodeClas.pt", help="save path for model. (default: MaskGAE-NodeClas.pt)")
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--full_data', action='store_true', help='Whether to use full data for pretraining. (default: False)')


try:
    args = parser.parse_args()
    print(tab_printer(args))
except:
    parser.print_help()
    exit(0)

set_seed(args.seed)

if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.ToUndirected(),
    T.ToDevice(device),
])


# (!IMPORTANT) Specify the path to your dataset directory ##############
# root = '~/public_data/pyg_data' # my root directory
root = 'data/'
########################################################################

if args.dataset != 'ppi':
    data = get_dataset(root, args.dataset, transform=transform)

    train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=False)(data)
else:
    (train_datasets, train_edge_lists), (valid_datasets, valid_edge_lists), (test_datasets, test_edge_lists), (train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes) = get_dataset(root, args.dataset, transform=None)
    tmp = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                            is_undirected=True,
                            split_labels=True,
                            add_negative_train_samples=False)(train_datasets[0])

if args.full_data:
    # Use full graph for pretraining
    splits = dict(train=data, valid=val_data, test=test_data)
else:
    splits = dict(train=train_data, valid=val_data, test=test_data)


if args.mask == 'Path':
    mask = MaskPath(p=args.p, 
                    num_nodes=data.num_nodes, 
                    start=args.start,
                    walk_length=args.encoder_layers+1)
elif args.mask == 'Edge':
    mask = MaskEdge(p=args.p)
else:
    mask = None # vanilla GAE

test_thin_and_thinner = True  # True False
test_fine_tune = False  # False True

node_number_dict = {
    'Cora': 140,
    'Citeseer': 8,
    'Pubmed': 8,
    'Photo': 118,
    'Computers': 212,
    'arxiv': 2048,
}
node_number = node_number_dict[args.dataset]

encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation)

edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                           num_layers=args.decoder_layers, dropout=args.decoder_dropout)

degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)


model = MaskGAE(encoder, edge_decoder, degree_decoder, mask, node_number=node_number).to(device)

# wzb edit
if test_thin_and_thinner:
    encoder_thin = GNNEncoder(data.num_features, args.encoder_channels // 2, args.hidden_channels // 2,
                        num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                        bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder_thin = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                            num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    degree_decoder_thin = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                                num_layers=args.decoder_layers, dropout=args.decoder_dropout)


    model_thin = MaskGAE(encoder_thin, edge_decoder_thin, degree_decoder_thin, mask, node_number=node_number).to(device)

    encoder_quarter = GNNEncoder(data.num_features, args.encoder_channels // 4, args.hidden_channels // 4,
                            num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                            bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder_quarter = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                                num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    degree_decoder_quarter = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                                    num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    model_quarter = MaskGAE(encoder_quarter, edge_decoder_quarter, degree_decoder_quarter, mask, node_number=node_number).to(device)

if not test_fine_tune:
    encoder_new = GNNEncoder(data.num_features, args.encoder_channels, min(args.hidden_channels, args.encoder_channels),
                            num_layers=2, dropout=args.encoder_dropout,
                            bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder_new = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                                num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    degree_decoder_new = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                                    num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    model_new = MaskGAE(encoder_new, edge_decoder_new, degree_decoder_new, mask, node_number=node_number).to(device)


encoder_new_thin = GNNEncoder(data.num_features, args.encoder_channels, min(args.hidden_channels // 2, args.encoder_channels),
                        num_layers=2, dropout=args.encoder_dropout,
                        bn=args.bn, layer=args.layer, activation=args.encoder_activation)

edge_decoder_new_thin = EdgeDecoder(args.hidden_channels // 2, args.decoder_channels,
                            num_layers=args.decoder_layers, dropout=args.decoder_dropout)

degree_decoder_new_thin = DegreeDecoder(args.hidden_channels // 2, args.decoder_channels,
                                num_layers=args.decoder_layers, dropout=args.decoder_dropout)

model_new_thin = MaskGAE(encoder_new_thin, edge_decoder_new_thin, degree_decoder_new_thin, mask, node_number=node_number).to(device)

if not test_fine_tune:
    encoder_new_quarter = GNNEncoder(data.num_features, args.encoder_channels, min(args.hidden_channels // 4, args.encoder_channels),
                            num_layers=2, dropout=args.encoder_dropout,
                            bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder_new_quarter = EdgeDecoder(args.hidden_channels // 4, args.decoder_channels,
                                num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    degree_decoder_new_quarter = DegreeDecoder(args.hidden_channels // 4, args.decoder_channels,
                                    num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    model_new_quarter = MaskGAE(encoder_new_quarter, edge_decoder_new_quarter, degree_decoder_new_quarter, mask, node_number=node_number).to(device)

print('encoder paramters', sum(p.numel() for p in encoder.parameters()))

if test_thin_and_thinner:
    print('encoder thin paramters', sum(p.numel() for p in encoder_thin.parameters()))
    print('encoder quarter paramters', sum(p.numel() for p in encoder_quarter.parameters()))

if not test_fine_tune:
    print('encoder new paramters', sum(p.numel() for p in encoder_new.parameters()))

print('encoder new thin paramters', sum(p.numel() for p in encoder_new_thin.parameters()))

if not test_fine_tune:
    print('encoder new quarter paramters', sum(p.numel() for p in encoder_new_quarter.parameters()))

train_linkpred(model, splits, args, device=device)

if args.dataset != 'arxiv' or not test_fine_tune:
    train_nodeclas(model, data, args, base=2, device=device)
else:
    train_nodeclas(model, data, args, base=1, device=device)

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
        if "head" in key:
            continue
        weight_selection[key] = uniform_element_selection(teacher_weights[key], student_weights[key].shape, device)
    return weight_selection

# wzb edit
if test_thin_and_thinner:
    model_state_dict = model.state_dict()

    teacher_weights = model_state_dict
    student_weights = model_thin.state_dict()
    model_thin.load_state_dict(get_slim_weight(teacher_weights, student_weights, device=device))
    train_nodeclas(model_thin, data, args, base=4, device=device)  # base 4

    teacher_weights = model_state_dict
    student_weights = model_quarter.state_dict()
    model_quarter.load_state_dict(get_slim_weight(teacher_weights, student_weights, device=device))
    train_nodeclas(model_quarter, data, args, base=6, device=device)  # base 6

if not test_fine_tune:
    teacher_weights = model_state_dict
    student_weights = model_new.state_dict()
    model_new.load_state_dict(get_slim_weight(teacher_weights, student_weights, device=device))
    train_nodeclas(model_new, data, args, base=4, device=device)  # base 4

teacher_weights = model.state_dict()
student_weights = model_new_thin.state_dict()
model_new_thin.load_state_dict(get_slim_weight(teacher_weights, student_weights, device=device))
if args.dataset != 'arxiv' or not test_fine_tune:
    train_nodeclas(model_new_thin, data, args, base=4, device=device)
else:
    train_nodeclas(model_new_thin, data, args, base=2, device=device)

if not test_fine_tune:
    teacher_weights = model_state_dict
    student_weights = model_new_quarter.state_dict()
    model_new_quarter.load_state_dict(get_slim_weight(teacher_weights, student_weights, device=device))
    train_nodeclas(model_new_quarter, data, args, base=6, device=device)

model_fine_tune_dict = {
    'Cora': {
        'lr': 1e-4,
        'weight_decay': 1e-3,
        'clf_lr': 5e-3,
        'clf_weight_decay': 1e-3,
    },
    'Citeseer': {
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'clf_lr': 0.01,
        'clf_weight_decay': 5e-3
    },
    'Pubmed': {
        'lr': 1e-4,
        'weight_decay': 0,
        'clf_lr': 0.015,
        'clf_weight_decay': 5e-4
    },
    'Photo': {
        'lr': 1e-4,
        'weight_decay': 0,
        'clf_lr': 0.01,
        'clf_weight_decay': 1e-2
    },
    'Computers': {
        'lr': 2e-4,
        'weight_decay': 0,
        'clf_lr': 0.005,
        'clf_weight_decay': 5e-3,
    },
    'arxiv': {
        'lr': 1e-4,
        'weight_decay': 0,
        'clf_lr': 0.01,
        'clf_weight_decay': 0
    },
}
model_epoch_dict = {
    'Cora': 400,
    'Citeseer': 400,
    'Pubmed': 300,
    'Photo': 200,
    'Computers': 200,
    'arxiv': 100,
}

model_new_thin_fine_tune_dict = {
    'Cora': {
        'lr': 1e-4,
        'weight_decay': 5e-3,
        'clf_lr': 0.05,
        'clf_weight_decay': 1e-3,
    },
    'Citeseer': {
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'clf_lr': 0.01,
        'clf_weight_decay': 1e-2
    },
    'Pubmed': {
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'clf_lr': 0.02,
        'clf_weight_decay': 1e-3 
    },
    'Photo': {
        'lr': 1e-5,
        'weight_decay': 0,
        'clf_lr': 0.01,
        'clf_weight_decay': 5e-3
    },
    'Computers': {
        'lr': 1e-4,
        'weight_decay': 0,
        'clf_lr': 0.005,
        'clf_weight_decay': 5e-3,
    },
    'arxiv': {
        'lr': 2e-4,
        'weight_decay': 0,
        'clf_lr': 3e-3,
        'clf_weight_decay': 0
    },
}
model_new_thin_epoch_dict = {
    'Cora': 400,
    'Citeseer': 400,
    'Pubmed': 300,
    'Photo': 200,
    'Computers': 200,
    'arxiv': 100,
}

model_new_thin_reweight_fine_tune_dict = {
    'Cora': {
        'lr': 1e-4,
        'weight_decay': 5e-3,
        'clf_lr': 0.05,
        'clf_weight_decay': 1e-3,
    },
    'Citeseer': {
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'clf_lr': 0.01,
        'clf_weight_decay': 1e-2
    },
    'Pubmed': {
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'clf_lr': 0.02,
        'clf_weight_decay': 1e-3
    },
    'Photo': {
        'lr': 1e-4,
        'weight_decay': 0,
        'clf_lr': 0.02,
        'clf_weight_decay': 6e-3
    },
    'Computers': {
        'lr': 1e-4,
        'weight_decay': 0,
        'clf_lr': 0.005,
        'clf_weight_decay': 5e-3,
    },
    'arxiv': {
        'lr': 2e-4,
        'weight_decay': 0,
        'clf_lr': 3e-3,
        'clf_weight_decay': 0,
    },
}

model_new_thin_reweight_epoch_dict = {
    'Cora': 400,
    'Citeseer': 200,
    'Pubmed': 200,
    'Photo': 200,
    'Computers': 200,
    'arxiv': 100,
}

if test_fine_tune:
    train_nodeclas_fine_tune(model_new_thin, data, args, model_learning_rate=model_new_thin_fine_tune_dict[args.dataset]['lr'], model_weight_decay=model_new_thin_fine_tune_dict[args.dataset]['weight_decay'], clf_learning_rate=model_new_thin_fine_tune_dict[args.dataset]['clf_lr'], clf_weight_decay=model_new_thin_fine_tune_dict[args.dataset]['clf_weight_decay'], epoch_num=model_new_thin_epoch_dict[args.dataset], device=device)
    train_nodeclas_fine_tune(model, data, args, model_learning_rate=model_fine_tune_dict[args.dataset]['lr'], model_weight_decay=model_fine_tune_dict[args.dataset]['weight_decay'], clf_learning_rate=model_fine_tune_dict[args.dataset]['clf_lr'], clf_weight_decay=model_fine_tune_dict[args.dataset]['clf_weight_decay'], epoch_num=model_epoch_dict[args.dataset], device=device)
    train_nodeclas_fine_tune(model_new_thin, data, args, model_learning_rate=model_new_thin_reweight_fine_tune_dict[args.dataset]['lr'], model_weight_decay=model_new_thin_reweight_fine_tune_dict[args.dataset]['weight_decay'], clf_learning_rate=model_new_thin_reweight_fine_tune_dict[args.dataset]['clf_lr'], clf_weight_decay=model_new_thin_reweight_fine_tune_dict[args.dataset]['clf_weight_decay'], epoch_num=model_new_thin_reweight_epoch_dict[args.dataset], device=device, reweight=True, batch_size_dict=node_number_dict)
