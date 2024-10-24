import argparse
import numpy as np

import torch
import torch_geometric.transforms as T
from tqdm.auto import tqdm

# custom modules
from maskgae.utils import set_seed, tab_printer, get_dataset
from maskgae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder, DotEdgeDecoder
from maskgae.mask import MaskEdge, MaskPath

# wzb edit
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

# wzb edit
def train_linkpred(model, splits, args, device="cpu", model_thin=None):
# def train_linkpred(model, splits, args, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_valid = 0
    batch_size = args.batch_size
    
    train_data = splits['train'].to(device)
    valid_data = splits['valid'].to(device)
    test_data = splits['test'].to(device)
    
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

    # wzb edit
    model_state_dict = model.state_dict()
    model_thin_state_dict = model_thin.state_dict()
    teacher_weights = model_state_dict
    student_weights = model_thin_state_dict
    model_thin.load_state_dict(get_slim_weight(teacher_weights, student_weights, device=device))

    test_auc, test_ap = model.test_step(test_data, 
                                        test_data.pos_edge_label_index, 
                                        test_data.neg_edge_label_index, 
                                        batch_size=batch_size)   

    test_auc_thin, test_ap_thin = model_thin.test_step(test_data, 
                                        test_data.pos_edge_label_index, 
                                        test_data.neg_edge_label_index, 
                                        batch_size=batch_size)   

    return test_auc, test_ap, test_auc_thin, test_ap_thin


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument("--mask", nargs="?", default="Path", help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=128, help='Channels of hidden representation. (default: 128)')
parser.add_argument('--decoder_channels', type=int, default=64, help='Channels of decoder. (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers of encoder. (default: 1)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.7)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.3)')
parser.add_argument('--alpha', type=float, default=0.003, help='loss weight for degree prediction. (default: 2e-3)')

parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size. (default: 2**16)')

parser.add_argument("--start", nargs="?", default="edge", help="Which Type to sample starting nodes for random walks, (default: edge)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. (default: 300)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=10, help='(default: 10)')
parser.add_argument("--save_path", nargs="?", default="MaskGAE-LinkPred.pt", help="save path for model. (default: MaskGAE-LinkPred.pt)")
parser.add_argument("--device", type=int, default=0)


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

data = get_dataset(root, args.dataset, transform=transform)

train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=False)(data)

splits = dict(train=train_data, valid=val_data, test=test_data)

if args.mask == 'Path':
    mask = MaskPath(p=args.p, num_nodes=data.num_nodes, 
                    start=args.start,
                    walk_length=args.encoder_layers+1)
elif args.mask == 'Edge':
    mask = MaskEdge(p=args.p)
else:
    mask = None # vanilla GAE

encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation)

if args.decoder_layers == 0:
    edge_decoder = DotEdgeDecoder()
else:
    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)

degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)


model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)

part_num = 2
# wzb edit
encoder_thin = GNNEncoder(data.num_features, args.encoder_channels // part_num, args.hidden_channels // part_num,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation)

if args.decoder_layers == 0:
    edge_decoder_thin = DotEdgeDecoder()
else:
    edge_decoder_thin = EdgeDecoder(args.hidden_channels // part_num, args.decoder_channels,  # decoder_channels 不要除以2，不然差的会多，就看预训练模型的冗余
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)

degree_decoder_thin = DegreeDecoder(args.hidden_channels // part_num, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)


model_thin = MaskGAE(encoder_thin, edge_decoder_thin, degree_decoder_thin, mask).to(device)

num_params = [p.numel() for p in model.parameters() if  p.requires_grad]
print(f"num parameters for model: {sum(num_params)}")

num_edge_decoder_params = [p.numel() for p in edge_decoder.parameters() if  p.requires_grad]
print(f"num parameters for edge_decoder: {sum(num_edge_decoder_params)}")

num_params_thin = [p.numel() for p in model_thin.parameters() if  p.requires_grad]
print(f"num parameters for model_thin: {sum(num_params_thin)}")

num_edge_decoder_params_thin = [p.numel() for p in edge_decoder_thin.parameters() if  p.requires_grad]
print(f"num parameters for edge_decoder_thin: {sum(num_edge_decoder_params_thin)}")

auc_results = []
ap_results = []

# wzb edit
auc_results_thin = []
ap_results_thin = []

for run in range(1, args.runs+1):
    # wzb edit
    test_auc, test_ap, test_auc_thin, test_ap_thin = train_linkpred(model, splits, args, device=device, model_thin=model_thin)

    # test_auc, test_ap = train_linkpred(model, splits, args, device=device)
    auc_results.append(test_auc)
    ap_results.append(test_ap)
    print(f'Runs {run} - AUC: {test_auc:.2%}', f'AP: {test_ap:.2%}')    

    # wzb edit
    auc_results_thin.append(test_auc_thin)
    ap_results_thin.append(test_ap_thin)
    print(f'Runs {run} - AUC_thin: {test_auc_thin:.2%}', f'AP_thin: {test_ap_thin:.2%}')

print(f'Link Prediction Results ({args.runs} runs):\n'
      f'AUC: {np.mean(auc_results):.2%} ± {np.std(auc_results):.2%}',
      f'AP: {np.mean(ap_results):.2%} ± {np.std(ap_results):.2%}',
     )

# wzb edit
print(f'Link Prediction Results ({args.runs} runs) Thin:\n'
      f'AUC_thin: {np.mean(auc_results_thin):.2%} ± {np.std(auc_results_thin):.2%}',
      f'AP_thin: {np.mean(ap_results_thin):.2%} ± {np.std(ap_results_thin):.2%}',
     )
