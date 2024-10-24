import logging
import numpy as np
from tqdm import tqdm
import torch

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset
from graphmae.evaluation import node_classification_evaluation
from graphmae.models import build_model, build_thin_model

import os
import sys
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from CKA import CudaCKA


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None, dataset_name=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 200 == 0:
            node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True, dataset_name=dataset_name)

    # return best_model
    return model

# wzb edit
def uniform_element_selection(wt, s_shape):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"  # determine whether teacher is larger than student on this dimension
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim] - 1, s_shape[dim]))
        indices = indices.to(torch.int64)
        indices = indices.to(ws.device)
        ws = torch.index_select(ws, dim, indices)
    assert ws.shape == s_shape
    return ws

# # wzb edit
# def uniform_element_selection(wt, s_shape, sjy_dict):
#     assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
#     ws = wt.clone()

#     dims = list(range(wt.dim()))
#     if len(dims) != 1:
#         for dim in range(wt.dim()):
#             assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"  # determine whether teacher is larger than student on this dimension
#             if s_shape[dim] in sjy_dict:
#                 indices = sjy_dict[s_shape[dim]]
#             else:

#                 if wt.shape[dim] % s_shape[dim] == 0:
#                     step = wt.shape[dim] // s_shape[dim]
#                     indices = torch.arange(s_shape[dim]) * step
#                 else:
#                     indices = torch.round(torch.linspace(0, wt.shape[dim] - 1, s_shape[dim]))
#                 print(wt.dim(), wt.shape, s_shape)
#                 print('indices1', indices)

#                 # I want to choose the maximum value in the teacher's dimension instead of indices
#                 dims1 = dims.copy()
#                 dims1.remove(dim)
#                 abs_mean = torch.mean(torch.abs(wt), dim=tuple(dims1))
#                 print('abs_mean', abs_mean.shape)
#                 _, indices = torch.topk(abs_mean, s_shape[dim], sorted=True)
#                 print('indices2', indices)
#                 # I want to sort indices
#                 indices, _ = torch.sort(indices)
#                 print('indices3', indices)

#             indices = indices.to(torch.int64)
#             ws = torch.index_select(ws, dim, indices)
#     else:
#         if s_shape[0] in sjy_dict:
#             indices = sjy_dict[s_shape[0]]
#         else:
#             _, indices = torch.topk(torch.abs(wt), s_shape[0], sorted=True)
#             sjy_dict[s_shape[0]] = indices
#         indices = indices.to(torch.int64)
#         indices, _ = torch.sort(indices)
#         ws = torch.index_select(ws, 0, indices)
#     assert ws.shape == s_shape
#     return sjy_dict, ws

def get_slim_weight(teacher_weights, student_weights):
    weight_selection = {}


    for key in student_weights.keys():
        if "head" in key:
            continue
        weight_selection[key] = uniform_element_selection(teacher_weights[key], student_weights[key].shape)
    return weight_selection

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []

    f1ma_list = []
    estp_f1ma_list = []

    # wzb edit
    thin_acc_list = []
    thin_estp_acc_list = []
    thin_f1ma_list = []
    thin_estp_f1ma_list = []
    thinner_acc_list = []
    thinner_estp_acc_list = []
    thinner_f1ma_list = []
    thinner_estp_f1ma_list = []
    thinner_not_thin_acc_list = []
    thinner_not_thin_estp_acc_list = []
    thinner_not_thin_f1ma_list = []
    thinner_not_thin_estp_f1ma_list = []
    thinner_only_thin_final_acc_list = []
    thinner_only_thin_final_estp_acc_list = []
    thinner_only_thin_final_f1ma_list = []
    thinner_only_thin_final_estp_f1ma_list = []

    test_thin_and_thinner = True  # False True
    new_thin_flag = True  # True True
    test_fine_tune = False  # True False

    other_thin = True  # True True
    by_thin = False  # False False

    if new_thin_flag:
        new_thin_acc_list = []
        new_thin_estp_acc_list = []
        new_thin_f1ma_list = []
        new_thin_estp_f1ma_list = []
        thin_new_thin_acc_list = []
        thin_new_thin_estp_acc_list = []
        thin_new_thin_f1ma_list = []
        thin_new_thin_estp_f1ma_list = []
        thinner_new_thin_acc_list = []
        thinner_new_thin_estp_acc_list = []
        thinner_new_thin_f1ma_list = []
        thinner_new_thin_estp_f1ma_list = []

    # wzb edit
    fat_tune_acc_list = []
    fat_tune_estp_acc_list = []
    fat_tune_f1ma_list = []
    fat_tune_estp_f1ma_list = []

    thin_new_thin_tune_acc_list = []
    thin_new_thin_tune_estp_acc_list = []
    thin_new_thin_tune_f1ma_list = []
    thin_new_thin_tune_estp_f1ma_list = []

    thin_new_thin_tune_reweight_acc_list = []
    thin_new_thin_tune_reweight_estp_acc_list = []
    thin_new_thin_tune_reweight_f1ma_list = []
    thin_new_thin_tune_reweight_estp_f1ma_list = []

    print('other_thin', other_thin)

    cka_max_node_num = 10000
    cka = CudaCKA(device=device)
    cka_score_array_between_models = {}
    cka_score_array_between_layers = {}

    batch_size_dict = {
        'cora': 16,
        'citeseer': 120,
        'pubmed': 60,
        'photo': 153,
        'computer': 275,
        'ogbn-arxiv': 90941, 
    }
    # node_number = graph.ndata["label"][graph.ndata["train_mask"]].shape[0]
    node_number = batch_size_dict[dataset_name]

    for i, seed in enumerate(seeds):
        checkpoint_path = f'./checkpoint/{dataset_name}'
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        model_save_path = f'./checkpoint/{dataset_name}/model_state_dict_{i}_{dataset_name}.pt'
        model_thin_new_thin_save_path = f'./checkpoint/{dataset_name}/model_thin_new_thin_state_dict_{i}_{dataset_name}.pt'

        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args, node_number=node_number)
        model.to(device)

        # wzb edit
        if test_thin_and_thinner:
            if not test_fine_tune:
                model_thin = build_model(args, if_thin=True, other_thin=other_thin, thin_base=2, node_number=node_number)
                model_thin = model_thin.to(device)
                model_thin_state_dict = model_thin.state_dict()

                model_thinner = build_model(args, if_thin=True, other_thin=other_thin, thin_base=4, node_number=node_number)
                model_thinner = model_thinner.to(device)
                model_thinner_state_dict = model_thinner.state_dict()

        if by_thin:
            model_thinner_not_thin_final = build_model(args, if_thin=True, other_thin=other_thin, thin_base=4, final_thin=False, node_number=node_number)
            model_thinner_not_thin_final = model_thinner_not_thin_final.to(device)
            model_thinner_not_thin_final_state_dict = model_thinner_not_thin_final.state_dict()

            model_thinner_only_thin_final = build_model(args, if_thin=True, other_thin=False, thin_base=4, node_number=node_number)
            model_thinner_only_thin_final = model_thinner_only_thin_final.to(device)
            model_thinner_only_thin_final_state_dict = model_thinner_only_thin_final.state_dict()

        if new_thin_flag:
            if not test_fine_tune:
                model_new_thin = build_model(args, if_thin=True, other_thin=other_thin, thin_base=1, thin_type='new_thin', node_number=node_number)
                model_new_thin = model_new_thin.to(device)
                model_new_thin_state_dict = model_new_thin.state_dict()

            model_thin_new_thin = build_model(args, if_thin=True, other_thin=other_thin, thin_base=2, thin_type='new_thin', node_number=node_number)
            model_thin_new_thin = model_thin_new_thin.to(device)
            model_thin_new_thin_state_dict = model_thin_new_thin.state_dict()

            if not test_fine_tune:
                model_thinner_new_thin = build_model(args, if_thin=True, other_thin=other_thin, thin_base=4, thin_type='new_thin', node_number=node_number)
                model_thinner_new_thin = model_thinner_new_thin.to(device)
                model_thinner_new_thin_state_dict = model_thinner_new_thin.state_dict()

        print('model encoder params', sum(p.numel() for p in model.encoder.parameters()))
        if test_thin_and_thinner:
            if not test_fine_tune:
                print('model thin encoder params', sum(p.numel() for p in model_thin.encoder.parameters()))
                print('model thinner encoder params', sum(p.numel() for p in model_thinner.encoder.parameters()))
        if by_thin:
            print('model thinner not thin final encoder params', sum(p.numel() for p in model_thinner_not_thin_final.encoder.parameters()))
            print('model thinner only thin final encoder params', sum(p.numel() for p in model_thinner_only_thin_final.encoder.parameters()))
        
        if new_thin_flag:
            if not test_fine_tune:
                print('model drop layers encoder params', sum(p.numel() for p in model_new_thin.encoder.parameters()))
            print('model thin drop layers encoder params', sum(p.numel() for p in model_thin_new_thin.encoder.parameters()))
            if not test_fine_tune:
                print('model thinner drop layers encoder params', sum(p.numel() for p in model_thinner_new_thin.encoder.parameters()))

        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, True, logger, dataset_name)
            model = model.cpu()
            torch.save(model.state_dict(), model_save_path)

        if load_model:  # False
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load(model_save_path))
        if save_model:  # False
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        # wzb edit
        model_state_dict = model.state_dict()
        # model linear probe
        model = model.to(device)
        model.eval()

        # wzb edit
        if test_thin_and_thinner:
            if not test_fine_tune:
                model_thin = model_thin.to(device)
                model_thin.eval()
                model_thinner = model_thinner.to(device)
                model_thinner.eval()

        if by_thin:
            model_thinner_not_thin_final = model_thinner_not_thin_final.to(device)
            model_thinner_not_thin_final.eval()
            model_thinner_only_thin_final = model_thinner_only_thin_final.to(device)
            model_thinner_only_thin_final.eval()

        if new_thin_flag:
            if not test_fine_tune:
                model_new_thin = model_new_thin.to(device)
                model_new_thin.eval()
            model_thin_new_thin = model_thin_new_thin.to(device)
            model_thin_new_thin.eval()
            if not test_fine_tune:
                model_thinner_new_thin = model_thinner_new_thin.to(device)
                model_thinner_new_thin.eval()

        if test_thin_and_thinner:
            if not test_fine_tune:
                teacher_weights = model_state_dict
                student_weights = model_thin_state_dict
                weight_selection = get_slim_weight(teacher_weights, student_weights)
                model_thin.load_state_dict(weight_selection)

                teacher_weights = model_state_dict
                student_weights = model_thinner_state_dict
                weight_selection = get_slim_weight(teacher_weights, student_weights)
                model_thinner.load_state_dict(weight_selection)

        if by_thin:
            teacher_weights = model_state_dict
            student_weights = model_thinner_not_thin_final_state_dict
            weight_selection = get_slim_weight(teacher_weights, student_weights)
            model_thinner_not_thin_final.load_state_dict(weight_selection)

            teacher_weights = model_state_dict
            student_weights = model_thinner_only_thin_final_state_dict
            weight_selection = get_slim_weight(teacher_weights, student_weights)
            model_thinner_only_thin_final.load_state_dict(weight_selection)

        if new_thin_flag:
            if not test_fine_tune:
                teacher_weights = model_state_dict
                student_weights = model_new_thin_state_dict
                weight_selection = get_slim_weight(teacher_weights, student_weights)
                model_new_thin.load_state_dict(weight_selection)

            teacher_weights = model_state_dict
            student_weights = model_thin_new_thin_state_dict
            weight_selection = get_slim_weight(teacher_weights, student_weights)
            model_thin_new_thin.load_state_dict(weight_selection)

            if not test_fine_tune:
                teacher_weights = model_state_dict
                student_weights = model_thinner_new_thin_state_dict
                weight_selection = get_slim_weight(teacher_weights, student_weights)
                model_thinner_new_thin.load_state_dict(weight_selection)

        (final_acc, estp_acc, final_f1ma, estp_f1ma), hidden_list = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, epoch_base=1, dataset_name=dataset_name)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)
        f1ma_list.append(final_f1ma)
        estp_f1ma_list.append(estp_f1ma)

        if test_thin_and_thinner:
            if not test_fine_tune:
                (thin_final_acc, thin_estp_acc, thin_final_f1ma, thin_estp_f1ma), thin_hidden_list = node_classification_evaluation(model_thin, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, epoch_base=1, dataset_name=dataset_name)
                thin_acc_list.append(thin_final_acc)
                thin_estp_acc_list.append(thin_estp_acc)
                thin_f1ma_list.append(thin_final_f1ma)
                thin_estp_f1ma_list.append(thin_estp_f1ma)

                (thinner_final_acc, thinner_estp_acc, thinner_final_f1ma, thinner_estp_f1ma), thinner_hidden_list = node_classification_evaluation(model_thinner, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, epoch_base=1, dataset_name=dataset_name)
                thinner_acc_list.append(thinner_final_acc)
                thinner_estp_acc_list.append(thinner_estp_acc)
                thinner_f1ma_list.append(thinner_final_f1ma)
                thinner_estp_f1ma_list.append(thinner_estp_f1ma)

        if by_thin:
            (thinner_not_thin_final_acc, thinner_not_thin_estp_acc, thinner_not_thin_final_f1ma, thinner_not_thin_estp_f1ma), thinner_not_thin_hidden_list = node_classification_evaluation(model_thinner_not_thin_final, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, epoch_base=1, dataset_name=dataset_name)
            thinner_not_thin_acc_list.append(thinner_not_thin_final_acc)
            thinner_not_thin_estp_acc_list.append(thinner_not_thin_estp_acc)
            thinner_not_thin_f1ma_list.append(thinner_not_thin_final_f1ma)
            thinner_not_thin_estp_f1ma_list.append(thinner_not_thin_estp_f1ma)

            (thinner_only_thin_final_acc, thinner_only_thin_estp_acc, thinner_only_thin_final_f1ma, thinner_only_thin_estp_f1ma), thinner_only_thin_hidden_list = node_classification_evaluation(model_thinner_only_thin_final, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, epoch_base=1, dataset_name=dataset_name)
            thinner_only_thin_final_acc_list.append(thinner_only_thin_final_acc)
            thinner_only_thin_final_estp_acc_list.append(thinner_only_thin_estp_acc)
            thinner_only_thin_final_f1ma_list.append(thinner_only_thin_final_f1ma)
            thinner_only_thin_final_estp_f1ma_list.append(thinner_only_thin_estp_f1ma)

        if new_thin_flag:
            if not test_fine_tune:
                (thin_new_final_acc, thin_new_estp_acc, thin_new_final_f1ma, thin_new_estp_f1ma), thin_new_hidden_list = node_classification_evaluation(model_new_thin, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, epoch_base=1, dataset_name=dataset_name)
                new_thin_acc_list.append(thin_new_final_acc)
                new_thin_estp_acc_list.append(thin_new_estp_acc)
                new_thin_f1ma_list.append(thin_new_final_f1ma)
                new_thin_estp_f1ma_list.append(thin_new_estp_f1ma)

            (thin_new_thin_final_acc, thin_new_thin_estp_acc, thin_new_thin_final_f1ma, thin_new_thin_estp_f1ma), thin_new_thin_hidden_list = node_classification_evaluation(model_thin_new_thin, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, epoch_base=1, dataset_name=dataset_name)
            thin_new_thin_acc_list.append(thin_new_thin_final_acc)
            thin_new_thin_estp_acc_list.append(thin_new_thin_estp_acc)
            thin_new_thin_f1ma_list.append(thin_new_thin_final_f1ma)
            thin_new_thin_estp_f1ma_list.append(thin_new_thin_estp_f1ma)

            if not test_fine_tune:
                (thinner_new_thin_final_acc, thinner_new_thin_estp_acc, thinner_new_thin_final_f1ma, thinner_new_thin_estp_f1ma), thinner_new_thin_hidden_list = node_classification_evaluation(model_thinner_new_thin, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, epoch_base=1, dataset_name=dataset_name)
                thinner_new_thin_acc_list.append(thinner_new_thin_final_acc)
                thinner_new_thin_estp_acc_list.append(thinner_new_thin_estp_acc)
                thinner_new_thin_f1ma_list.append(thinner_new_thin_final_f1ma)
                thinner_new_thin_estp_f1ma_list.append(thinner_new_thin_estp_f1ma)

        if test_thin_and_thinner:
            if not test_fine_tune:

                for cur_layer in range(len(hidden_list)):
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].to(device), hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                        if 'model' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model'][0].append(cka_score_between_data_and_layer0)
                    if cur_layer != len(hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :], hidden_list[cur_layer + 1][:cka_max_node_num, :]).cpu()
                        if 'model' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model']) < cur_layer + 2:
                                cka_score_array_between_layers['model'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model'][cur_layer + 1].append(cka_score_between_layer)

                for cur_layer in range(len(hidden_list)):
                    cka_score_between_model_and_thin = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :], thin_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                    if 'model_thin' not in cka_score_array_between_models:
                        cka_score_array_between_models['model_thin'] = [[cka_score_between_model_and_thin]]
                    else:
                        if len(cka_score_array_between_models['model_thin']) < cur_layer + 1:
                            cka_score_array_between_models['model_thin'].append([cka_score_between_model_and_thin])
                        else:
                            cka_score_array_between_models['model_thin'][cur_layer].append(cka_score_between_model_and_thin)
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].to(device), thin_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                        if 'model_thin' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_thin'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model_thin'][0].append(cka_score_between_data_and_layer0)
                    
                    if cur_layer != len(hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(thin_hidden_list[cur_layer][:cka_max_node_num, :], thin_hidden_list[cur_layer + 1][:cka_max_node_num, :]).cpu()
                        if 'model_thin' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_thin'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model_thin']) < cur_layer + 2:
                                cka_score_array_between_layers['model_thin'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model_thin'][cur_layer + 1].append(cka_score_between_layer)

                for cur_layer in range(len(hidden_list)):
                    cka_score_between_model_and_thinner = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :], thinner_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                    if 'model_thinner' not in cka_score_array_between_models:
                        cka_score_array_between_models['model_thinner'] = [[cka_score_between_model_and_thinner]]
                    else:
                        if len(cka_score_array_between_models['model_thinner']) < cur_layer + 1:
                            cka_score_array_between_models['model_thinner'].append([cka_score_between_model_and_thinner])
                        else:
                            cka_score_array_between_models['model_thinner'][cur_layer].append(cka_score_between_model_and_thinner)
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].to(device), thinner_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                        if 'model_thinner' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_thinner'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model_thinner'][0].append(cka_score_between_data_and_layer0)
                    
                    if cur_layer != len(hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(thinner_hidden_list[cur_layer][:cka_max_node_num, :], thinner_hidden_list[cur_layer + 1][:cka_max_node_num, :]).cpu()
                        if 'model_thinner' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_thinner'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model_thinner']) < cur_layer + 2:
                                cka_score_array_between_layers['model_thinner'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model_thinner'][cur_layer + 1].append(cka_score_between_layer)

        if by_thin:
            for cur_layer in range(len(hidden_list)):
                cka_score_between_model_and_thinner_not_thin = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :], thinner_not_thin_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                if len(cka_score_array_between_models) < 3:
                    cka_score_array_between_models.append([[cka_score_between_model_and_thinner_not_thin]])
                else:
                    if len(cka_score_array_between_models[2]) < cur_layer + 1:
                        cka_score_array_between_models[2].append([cka_score_between_model_and_thinner_not_thin])
                    else:
                        cka_score_array_between_models[2][cur_layer].append(cka_score_between_model_and_thinner_not_thin)
                if cur_layer == 0:
                    cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].to(device), thinner_not_thin_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                    if len(cka_score_array_between_layers) < 4:
                        cka_score_array_between_layers.append([[cka_score_between_data_and_layer0]])
                    else:
                        if len(cka_score_array_between_layers[3]) < 1:
                            cka_score_array_between_layers[3].append([cka_score_between_data_and_layer0])
                        else:
                            cka_score_array_between_layers[3][0].append(cka_score_between_data_and_layer0)
                
                if cur_layer != len(hidden_list) - 1:
                    cka_score_between_layer = cka.linear_CKA(thinner_not_thin_hidden_list[cur_layer][:cka_max_node_num, :], thinner_not_thin_hidden_list[cur_layer + 1][:cka_max_node_num, :]).cpu()
                    if len(cka_score_array_between_layers) < 4:
                        cka_score_array_between_layers.append([[cka_score_between_layer]])
                    else:
                        if len(cka_score_array_between_layers[3]) < cur_layer + 2:
                            cka_score_array_between_layers[3].append([cka_score_between_layer])
                        else:
                            cka_score_array_between_layers[3][cur_layer + 1].append(cka_score_between_layer)

            for cur_layer in range(len(hidden_list)):
                cka_score_between_model_and_thinner_only_thin = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :], thinner_only_thin_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                if len(cka_score_array_between_models) < 4:
                    cka_score_array_between_models.append([[cka_score_between_model_and_thinner_only_thin]])
                else:
                    if len(cka_score_array_between_models[3]) < cur_layer + 1:
                        cka_score_array_between_models[3].append([cka_score_between_model_and_thinner_only_thin])
                    else:
                        cka_score_array_between_models[3][cur_layer].append(cka_score_between_model_and_thinner_only_thin)
                if cur_layer == 0:
                    cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].to(device), thinner_only_thin_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                    if len(cka_score_array_between_layers) < 5:
                        cka_score_array_between_layers.append([[cka_score_between_data_and_layer0]])
                    else:
                        if len(cka_score_array_between_layers[4]) < 1:
                            cka_score_array_between_layers[4].append([cka_score_between_data_and_layer0])
                        else:
                            cka_score_array_between_layers[4][0].append(cka_score_between_data_and_layer0)
                
                if cur_layer != len(hidden_list) - 1:
                    cka_score_between_layer = cka.linear_CKA(thinner_only_thin_hidden_list[cur_layer][:cka_max_node_num, :], thinner_only_thin_hidden_list[cur_layer + 1][:cka_max_node_num, :]).cpu()
                    if len(cka_score_array_between_layers) < 5:
                        cka_score_array_between_layers.append([[cka_score_between_layer]])
                    else:
                        if len(cka_score_array_between_layers[4]) < cur_layer + 2:
                            cka_score_array_between_layers[4].append([cka_score_between_layer])
                        else:
                            cka_score_array_between_layers[4][cur_layer + 1].append(cka_score_between_layer)

        if not test_fine_tune:
            if new_thin_flag:
                for cur_layer in range(len(thin_new_hidden_list)):
                    cka_score_between_model_and_new_thin = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :], thin_new_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                    if 'model_new' not in cka_score_array_between_models:
                        cka_score_array_between_models['model_new'] = [[cka_score_between_model_and_new_thin]]
                    else:
                        if len(cka_score_array_between_models['model_new']) < cur_layer + 1:
                            cka_score_array_between_models['model_new'].append([cka_score_between_model_and_new_thin])
                        else:
                            cka_score_array_between_models['model_new'][cur_layer].append(cka_score_between_model_and_new_thin)
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].to(device), thin_new_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                        if 'model_new' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model_new'][0].append(cka_score_between_data_and_layer0)
                    
                    if cur_layer != len(thin_new_hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(thin_new_hidden_list[cur_layer][:cka_max_node_num, :], thin_new_hidden_list[cur_layer + 1][:cka_max_node_num, :]).cpu()
                        if 'model_new' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model_new']) < cur_layer + 2:
                                cka_score_array_between_layers['model_new'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model_new'][cur_layer + 1].append(cka_score_between_layer)

                for cur_layer in range(len(thin_new_thin_hidden_list)):
                    cka_score_between_model_and_thin_new_thin = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :], thin_new_thin_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                    if 'model_new_thin' not in cka_score_array_between_models:
                        cka_score_array_between_models['model_new_thin'] = [[cka_score_between_model_and_thin_new_thin]]
                    else:
                        if len(cka_score_array_between_models['model_new_thin']) < cur_layer + 1:
                            cka_score_array_between_models['model_new_thin'].append([cka_score_between_model_and_thin_new_thin])
                        else:
                            cka_score_array_between_models['model_new_thin'][cur_layer].append(cka_score_between_model_and_thin_new_thin)
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].to(device), thin_new_thin_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                        if 'model_new_thin' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new_thin'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model_new_thin'][0].append(cka_score_between_data_and_layer0)
                    
                    if cur_layer != len(thin_new_thin_hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(thin_new_thin_hidden_list[cur_layer][:cka_max_node_num, :], thin_new_thin_hidden_list[cur_layer + 1][:cka_max_node_num, :]).cpu()
                        if 'model_new_thin' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new_thin'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model_new_thin']) < cur_layer + 2:
                                cka_score_array_between_layers['model_new_thin'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model_new_thin'][cur_layer + 1].append(cka_score_between_layer)

                for cur_layer in range(len(thinner_new_thin_hidden_list)):
                    cka_score_between_model_and_thinner_new_thin = cka.linear_CKA(hidden_list[cur_layer][:cka_max_node_num, :], thinner_new_thin_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                    if 'model_new_thinner' not in cka_score_array_between_models:
                        cka_score_array_between_models['model_new_thinner'] = [[cka_score_between_model_and_thinner_new_thin]]
                    else:
                        if len(cka_score_array_between_models['model_new_thinner']) < cur_layer + 1:
                            cka_score_array_between_models['model_new_thinner'].append([cka_score_between_model_and_thinner_new_thin])
                        else:
                            cka_score_array_between_models['model_new_thinner'][cur_layer].append(cka_score_between_model_and_thinner_new_thin)
                    if cur_layer == 0:
                        cka_score_between_data_and_layer0 = cka.linear_CKA(x[:cka_max_node_num, :].to(device), thinner_new_thin_hidden_list[cur_layer][:cka_max_node_num, :]).cpu()
                        if 'model_new_thinner' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new_thinner'] = [[cka_score_between_data_and_layer0]]
                        else:
                            cka_score_array_between_layers['model_new_thinner'][0].append(cka_score_between_data_and_layer0)
                    
                    if cur_layer != len(thinner_new_thin_hidden_list) - 1:
                        cka_score_between_layer = cka.linear_CKA(thinner_new_thin_hidden_list[cur_layer][:cka_max_node_num, :], thinner_new_thin_hidden_list[cur_layer + 1][:cka_max_node_num, :]).cpu()
                        if 'model_new_thinner' not in cka_score_array_between_layers:
                            cka_score_array_between_layers['model_new_thinner'] = [[cka_score_between_layer]]
                        else:
                            if len(cka_score_array_between_layers['model_new_thinner']) < cur_layer + 2:
                                cka_score_array_between_layers['model_new_thinner'].append([cka_score_between_layer])
                            else:
                                cka_score_array_between_layers['model_new_thinner'][cur_layer + 1].append(cka_score_between_layer)
        re_del = True
        if test_fine_tune:
            citeseer_fine_tune_epoch = 250
            # wzb edit
            # new thin fine tune
            teacher_weights = model_state_dict
            student_weights = model_thin_new_thin_state_dict
            torch.save(student_weights, model_thin_new_thin_save_path)
            model_thin_new_thin.load_state_dict(get_slim_weight(teacher_weights, student_weights))
            model_thin_new_thin = model_thin_new_thin.to(device)
            model_thin_new_thin.train()
            if dataset_name != 'citeseer':
                (thin_new_thin_tune_final_acc, thin_new_thin_tune_estp_acc, thin_new_thin_tune_final_f1ma, thin_new_thin_tune_estp_f1ma), thin_new_thin_tune_hidden_list = node_classification_evaluation(model_thin_new_thin, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f * 1, device, False, dataset_name=dataset_name)
            else:
                (thin_new_thin_tune_final_acc, thin_new_thin_tune_estp_acc, thin_new_thin_tune_final_f1ma, thin_new_thin_tune_estp_f1ma), thin_new_thin_tune_hidden_list = node_classification_evaluation(model_thin_new_thin, graph, x, num_classes, lr_f, weight_decay_f, citeseer_fine_tune_epoch, device, False, dataset_name=dataset_name)
            thin_new_thin_tune_acc_list.append(thin_new_thin_tune_final_acc)
            thin_new_thin_tune_estp_acc_list.append(thin_new_thin_tune_estp_acc)
            thin_new_thin_tune_f1ma_list.append(thin_new_thin_tune_final_f1ma)
            thin_new_thin_tune_estp_f1ma_list.append(thin_new_thin_tune_estp_f1ma)
            
            if re_del:
                del model_thin_new_thin

            # wzb edit
            # new thin fine tune with reweight
            model_thin_new_thin = build_model(args, if_thin=True, other_thin=other_thin, thin_base=2, thin_type='new_thin', node_number=node_number)
            model_thin_new_thin.load_state_dict(get_slim_weight(model_state_dict, torch.load(model_thin_new_thin_save_path)))
            model_thin_new_thin = model_thin_new_thin.to(device)
            model_thin_new_thin.train()
            if dataset_name != 'citeseer':
                if dataset_name == 'cora':
                    (thin_new_thin_tune_reweight_final_acc, thin_new_thin_tune_reweight_estp_acc, thin_new_thin_tune_reweight_final_f1ma, thin_new_thin_tune_reweight_estp_f1ma), thin_new_thin_tune_reweight_hidden_list = node_classification_evaluation(model_thin_new_thin, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f // 6, device, False, reweight=True, dataset_name=dataset_name)
                elif dataset_name == 'photo':
                    (thin_new_thin_tune_reweight_final_acc, thin_new_thin_tune_reweight_estp_acc, thin_new_thin_tune_reweight_final_f1ma, thin_new_thin_tune_reweight_estp_f1ma), thin_new_thin_tune_reweight_hidden_list = node_classification_evaluation(model_thin_new_thin, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f * 1, device, False, reweight=True, dataset_name=dataset_name)
                else:
                    (thin_new_thin_tune_reweight_final_acc, thin_new_thin_tune_reweight_estp_acc, thin_new_thin_tune_reweight_final_f1ma, thin_new_thin_tune_reweight_estp_f1ma), thin_new_thin_tune_reweight_hidden_list = node_classification_evaluation(model_thin_new_thin, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f * 1, device, False, reweight=True, dataset_name=dataset_name)
            else:
                (thin_new_thin_tune_reweight_final_acc, thin_new_thin_tune_reweight_estp_acc, thin_new_thin_tune_reweight_final_f1ma, thin_new_thin_tune_reweight_estp_f1ma), thin_new_thin_tune_reweight_hidden_list = node_classification_evaluation(model_thin_new_thin, graph, x, num_classes, lr_f, weight_decay_f, citeseer_fine_tune_epoch, device, False, reweight=True, dataset_name=dataset_name)
            thin_new_thin_tune_reweight_acc_list.append(thin_new_thin_tune_reweight_final_acc)
            thin_new_thin_tune_reweight_estp_acc_list.append(thin_new_thin_tune_reweight_estp_acc)
            thin_new_thin_tune_reweight_f1ma_list.append(thin_new_thin_tune_reweight_final_f1ma)
            thin_new_thin_tune_reweight_estp_f1ma_list.append(thin_new_thin_tune_reweight_estp_f1ma)

            if re_del:
                del model_thin_new_thin

            # wzb edit
            # fat fine tune
            # model = model.to(device)
            if dataset_name != 'ogbn-arxiv':
                model.train()
                if dataset_name != 'citeseer':
                    (fat_tune_final_acc, fat_tune_estp_acc, fat_tune_final_f1ma, fat_tune_estp_f1ma), fat_tune_hidden_list = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f * 1, device, False, dataset_name=dataset_name)
                else:
                    (fat_tune_final_acc, fat_tune_estp_acc, fat_tune_final_f1ma, fat_tune_estp_f1ma), fat_tune_hidden_list = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, citeseer_fine_tune_epoch, device, False, dataset_name=dataset_name)
                fat_tune_acc_list.append(fat_tune_final_acc)
                fat_tune_estp_acc_list.append(fat_tune_estp_acc)
                fat_tune_f1ma_list.append(fat_tune_final_f1ma)
                fat_tune_estp_f1ma_list.append(fat_tune_estp_f1ma)

            if re_del:
                del model


        if logger is not None:
            logger.finish()
        
        if not re_del:
            del model
        if test_thin_and_thinner:
            if not test_fine_tune:
                del model_thin
                del model_thinner
        if by_thin:
            del model_thinner_not_thin_final
            del model_thinner_only_thin_final
        if new_thin_flag:
            if not test_fine_tune:
                del model_new_thin
            if not re_del:
                del model_thin_new_thin
            if not test_fine_tune:
                del model_thinner_new_thin

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    print(f'# final f1ma: {np.mean(f1ma_list):.4f}±{np.std(f1ma_list):.4f}')
    print(f'# early-stopping_f1ma: {np.mean(estp_f1ma_list):.4f}±{np.std(estp_f1ma_list):.4f}')

    if test_thin_and_thinner:
        if not test_fine_tune:
            thin_final_acc, thin_final_acc_std = np.mean(thin_acc_list), np.std(thin_acc_list)
            thin_estp_acc, thin_estp_acc_std = np.mean(thin_estp_acc_list), np.std(thin_estp_acc_list)
            print(f"# thin_final_acc: {thin_final_acc:.4f}±{thin_final_acc_std:.4f}")
            print(f"# thin_early-stopping_acc: {thin_estp_acc:.4f}±{thin_estp_acc_std:.4f}")
            print(f'# thin_final_f1ma: {np.mean(thin_f1ma_list):.4f}±{np.std(thin_f1ma_list):.4f}')
            print(f'# thin_early-stopping_f1ma: {np.mean(thin_estp_f1ma_list):.4f}±{np.std(thin_estp_f1ma_list):.4f}')

            thinner_final_acc, thinner_final_acc_std = np.mean(thinner_acc_list), np.std(thinner_acc_list)
            thinner_estp_acc, thinner_estp_acc_std = np.mean(thinner_estp_acc_list), np.std(thinner_estp_acc_list)
            print(f"# thinner_final_acc: {thinner_final_acc:.4f}±{thinner_final_acc_std:.4f}")
            print(f"# thinner_early-stopping_acc: {thinner_estp_acc:.4f}±{thinner_estp_acc_std:.4f}")
            print(f'# thinner_final_f1ma: {np.mean(thinner_f1ma_list):.4f}±{np.std(thinner_f1ma_list):.4f}')
            print(f'# thinner_early-stopping_f1ma: {np.mean(thinner_estp_f1ma_list):.4f}±{np.std(thinner_estp_f1ma_list):.4f}')

    if by_thin:
        thinner_not_thin_final_acc, thinner_not_thin_final_acc_std = np.mean(thinner_not_thin_acc_list), np.std(thinner_not_thin_acc_list)
        thinner_not_thin_estp_acc, thinner_not_thin_estp_acc_std = np.mean(thinner_not_thin_estp_acc_list), np.std(thinner_not_thin_estp_acc_list)
        print(f"# thinner_not_thin_final_acc: {thinner_not_thin_final_acc:.4f}±{thinner_not_thin_final_acc_std:.4f}")
        print(f"# thinner_not_thin_early-stopping_acc: {thinner_not_thin_estp_acc:.4f}±{thinner_not_thin_estp_acc_std:.4f}")
        print(f'# thinner_not_thin_final_f1ma: {np.mean(thinner_not_thin_f1ma_list):.4f}±{np.std(thinner_not_thin_f1ma_list):.4f}')
        print(f'# thinner_not_thin_early-stopping_f1ma: {np.mean(thinner_not_thin_estp_f1ma_list):.4f}±{np.std(thinner_not_thin_estp_f1ma_list):.4f}')

        thinner_only_thin_final_acc, thinner_only_thin_final_acc_std = np.mean(thinner_only_thin_final_acc_list), np.std(thinner_only_thin_final_acc_list)
        thinner_only_thin_estp_acc, thinner_only_thin_estp_acc_std = np.mean(thinner_only_thin_final_estp_acc_list), np.std(thinner_only_thin_final_estp_acc_list)
        print(f"# thinner_only_thin_final_acc: {thinner_only_thin_final_acc:.4f}±{thinner_only_thin_final_acc_std:.4f}")
        print(f"# thinner_only_thin_early-stopping_acc: {thinner_only_thin_estp_acc:.4f}±{thinner_only_thin_estp_acc_std:.4f}")
        print(f'# thinner_only_thin_final_f1ma: {np.mean(thinner_only_thin_final_f1ma_list):.4f}±{np.std(thinner_only_thin_final_f1ma_list):.4f}')
        print(f'# thinner_only_thin_early-stopping_f1ma: {np.mean(thinner_only_thin_final_estp_f1ma_list):.4f}±{np.std(thinner_only_thin_final_estp_f1ma_list):.4f}')

    if new_thin_flag:
        if not test_fine_tune:
            new_thin_final_acc, new_thin_final_acc_std = np.mean(new_thin_acc_list), np.std(new_thin_acc_list)
            new_thin_estp_acc, new_thin_estp_acc_std = np.mean(new_thin_estp_acc_list), np.std(new_thin_estp_acc_list)
            print(f"# new_thin_final_acc: {new_thin_final_acc:.4f}±{new_thin_final_acc_std:.4f}")
            print(f"# new_thin_early-stopping_acc: {new_thin_estp_acc:.4f}±{new_thin_estp_acc_std:.4f}")
            print(f'# new_thin_final_f1ma: {np.mean(new_thin_f1ma_list):.4f}±{np.std(new_thin_f1ma_list):.4f}')
            print(f'# new_thin_early-stopping_f1ma: {np.mean(new_thin_estp_f1ma_list):.4f}±{np.std(new_thin_estp_f1ma_list):.4f}')

        thin_new_thin_final_acc, thin_new_thin_final_acc_std = np.mean(thin_new_thin_acc_list), np.std(thin_new_thin_acc_list)
        thin_new_thin_estp_acc, thin_new_thin_estp_acc_std = np.mean(thin_new_thin_estp_acc_list), np.std(thin_new_thin_estp_acc_list)
        print(f"# thin_new_thin_final_acc: {thin_new_thin_final_acc:.4f}±{thin_new_thin_final_acc_std:.4f}")
        print(f"# thin_new_thin_early-stopping_acc: {thin_new_thin_estp_acc:.4f}±{thin_new_thin_estp_acc_std:.4f}")
        print(f'# thin_new_thin_final_f1ma: {np.mean(thin_new_thin_f1ma_list):.4f}±{np.std(thin_new_thin_f1ma_list):.4f}')
        print(f'# thin_new_thin_early-stopping_f1ma: {np.mean(thin_new_thin_estp_f1ma_list):.4f}±{np.std(thin_new_thin_estp_f1ma_list):.4f}')

        if not test_fine_tune:
            thinner_new_thin_final_acc, thinner_new_thin_final_acc_std = np.mean(thinner_new_thin_acc_list), np.std(thinner_new_thin_acc_list)
            thinner_new_thin_estp_acc, thinner_new_thin_estp_acc_std = np.mean(thinner_new_thin_estp_acc_list), np.std(thinner_new_thin_estp_acc_list)
            print(f"# thinner_new_thin_final_acc: {thinner_new_thin_final_acc:.4f}±{thinner_new_thin_final_acc_std:.4f}")
            print(f"# thinner_new_thin_early-stopping_acc: {thinner_new_thin_estp_acc:.4f}±{thinner_new_thin_estp_acc_std:.4f}")
            print(f'# thinner_new_thin_final_f1ma: {np.mean(thinner_new_thin_f1ma_list):.4f}±{np.std(thinner_new_thin_f1ma_list):.4f}')
            print(f'# thinner_new_thin_early-stopping_f1ma: {np.mean(thinner_new_thin_estp_f1ma_list):.4f}±{np.std(thinner_new_thin_estp_f1ma_list):.4f}')

    if test_thin_and_thinner:
        if not test_fine_tune:
            for i in cka_score_array_between_models:
                for j in range(len(cka_score_array_between_models[i])):
                    print(f'cka_score_between_models_{i}_layer_{j}: {np.mean(cka_score_array_between_models[i][j]):.4f}±{np.std(cka_score_array_between_models[i][j]):.4f}')

            for i in cka_score_array_between_layers:
                for j in range(len(cka_score_array_between_layers[i])):
                    print(f'cka_score_between_models_{i}_layer_{j}_and_layer_{j+1}: {np.mean(cka_score_array_between_layers[i][j]):.4f}±{np.std(cka_score_array_between_layers[i][j]):.4f}')

    if test_fine_tune:
        fat_tune_final_acc, fat_tune_final_acc_std = np.mean(fat_tune_acc_list), np.std(fat_tune_acc_list)
        fat_tune_estp_acc, fat_tune_estp_acc_std = np.mean(fat_tune_estp_acc_list), np.std(fat_tune_estp_acc_list)
        print(f"# fat_tune_final_acc: {fat_tune_final_acc:.4f}±{fat_tune_final_acc_std:.4f}")
        print(f"# fat_tune_early-stopping_acc: {fat_tune_estp_acc:.4f}±{fat_tune_estp_acc_std:.4f}")
        print(f'# fat_tune_final_f1ma: {np.mean(fat_tune_f1ma_list):.4f}±{np.std(fat_tune_f1ma_list):.4f}')
        print(f'# fat_tune_early-stopping_f1ma: {np.mean(fat_tune_estp_f1ma_list):.4f}±{np.std(fat_tune_estp_f1ma_list):.4f}')

        thin_new_thin_tune_final_acc, thin_new_thin_tune_final_acc_std = np.mean(thin_new_thin_tune_acc_list), np.std(thin_new_thin_tune_acc_list)
        thin_new_thin_tune_estp_acc, thin_new_thin_tune_estp_acc_std = np.mean(thin_new_thin_tune_estp_acc_list), np.std(thin_new_thin_tune_estp_acc_list)
        print(f"# thin_new_thin_tune_final_acc: {thin_new_thin_tune_final_acc:.4f}±{thin_new_thin_tune_final_acc_std:.4f}")
        print(f"# thin_new_thin_tune_early-stopping_acc: {thin_new_thin_tune_estp_acc:.4f}±{thin_new_thin_tune_estp_acc_std:.4f}")
        print(f'# thin_new_thin_tune_final_f1ma: {np.mean(thin_new_thin_tune_f1ma_list):.4f}±{np.std(thin_new_thin_tune_f1ma_list):.4f}')
        print(f'# thin_new_thin_tune_early-stopping_f1ma: {np.mean(thin_new_thin_tune_estp_f1ma_list):.4f}±{np.std(thin_new_thin_tune_estp_f1ma_list):.4f}')

        thin_new_thin_tune_reweight_final_acc, thin_new_thin_tune_reweight_final_acc_std = np.mean(thin_new_thin_tune_reweight_acc_list), np.std(thin_new_thin_tune_reweight_acc_list)
        thin_new_thin_tune_reweight_estp_acc, thin_new_thin_tune_reweight_estp_acc_std = np.mean(thin_new_thin_tune_reweight_estp_acc_list), np.std(thin_new_thin_tune_reweight_estp_acc_list)
        print(f"# thin_new_thin_tune_reweight_final_acc: {thin_new_thin_tune_reweight_final_acc:.4f}±{thin_new_thin_tune_reweight_final_acc_std:.4f}")
        print(f"# thin_new_thin_tune_reweight_early-stopping_acc: {thin_new_thin_tune_reweight_estp_acc:.4f}±{thin_new_thin_tune_reweight_estp_acc_std:.4f}")
        print(f'# thin_new_thin_tune_reweight_final_f1ma: {np.mean(thin_new_thin_tune_reweight_f1ma_list):.4f}±{np.std(thin_new_thin_tune_reweight_f1ma_list):.4f}')
        print(f'# thin_new_thin_tune_reweight_early-stopping_f1ma: {np.mean(thin_new_thin_tune_reweight_estp_f1ma_list):.4f}±{np.std(thin_new_thin_tune_reweight_estp_f1ma_list):.4f}')

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
