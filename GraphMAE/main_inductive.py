
import numpy as np
import torch
from sklearn.metrics import f1_score

import logging
import yaml
import numpy as np
from tqdm import tqdm
import torch

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
)
from graphmae.datasets.data_util import load_inductive_dataset
from graphmae.models import build_model, build_thin_model
from graphmae.evaluation import linear_probing_for_inductive_node_classiifcation, LogisticRegression

from main_transductive import uniform_element_selection, get_slim_weight

def evaluete(model, loaders, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        if len(loaders[0]) > 1:
            x_all = {"train": [], "val": [], "test": []}
            y_all = {"train": [], "val": [], "test": []}

            all_hiddens = []

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for ind, subgraph in enumerate(loader):
                        subgraph = subgraph.to(device)
                        feat = subgraph.ndata["feat"]
                        x, hiddens = model.embed(subgraph, feat, return_hidden=True)
                        x_all[key].append(x)
                        y_all[key].append(subgraph.ndata["label"])  
                        if ind == 0:
                            all_hiddens.append(hiddens)
            in_dim = x_all["train"][0].shape[1]
            encoder = LogisticRegression(in_dim, num_classes)
            num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
            if not mute:
                print(f"num parameters for finetuning: {sum(num_finetune_params)}")
                # torch.save(x.cpu(), "feat.pt")
            
            encoder.to(device)
            optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
            final_acc, estp_acc, final_f1ma, estp_f1ma = mutli_graph_linear_evaluation(encoder, x_all, y_all, optimizer_f, max_epoch_f, device, mute)
            return (final_acc, estp_acc, final_f1ma, estp_f1ma), all_hiddens
        else:
            x_all = {"train": None, "val": None, "test": None}
            y_all = {"train": None, "val": None, "test": None}

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for subgraph in loader:
                        subgraph = subgraph.to(device)
                        feat = subgraph.ndata["feat"]
                        x = model.embed(subgraph, feat)
                        mask = subgraph.ndata[f"{key}_mask"]
                        x_all[key] = x[mask]
                        y_all[key] = subgraph.ndata["label"][mask]  
            in_dim = x_all["train"].shape[1]
            
            encoder = LogisticRegression(in_dim, num_classes)
            encoder = encoder.to(device)
            optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)

            x = torch.cat(list(x_all.values()))
            y = torch.cat(list(y_all.values()))
            num_train, num_val, num_test = [x.shape[0] for x in x_all.values()]
            num_nodes = num_train + num_val + num_test
            train_mask = torch.arange(num_train, device=device)
            val_mask = torch.arange(num_train, num_train + num_val, device=device)
            test_mask = torch.arange(num_train + num_val, num_nodes, device=device)
            
            final_acc, estp_acc, final_f1ma, estp_f1ma = linear_probing_for_inductive_node_classiifcation(encoder, x, y, (train_mask, val_mask, test_mask), optimizer_f, max_epoch_f, device, mute)
            return final_acc, estp_acc, final_f1ma, estp_f1ma
    else:
        raise NotImplementedError


def mutli_graph_linear_evaluation(model, feat, labels, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0
    best_val_epoch = 0
    best_val_test_acc = 0
    best_val_f1ma = 0
    best_val_test_f1ma = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        for x, y in zip(feat["train"], labels["train"]):
            out = model(None, x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_out = []
            test_out = []
            for x, y in zip(feat["val"], labels["val"]):
                val_pred = model(None, x)
                val_out.append(val_pred)
            val_out = torch.cat(val_out, dim=0).cpu().numpy()
            val_label = torch.cat(labels["val"], dim=0).cpu().numpy()
            val_out = np.where(val_out >= 0, 1, 0)

            for x, y in zip(feat["test"], labels["test"]):
                test_pred = model(None, x)# 
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

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc}, test_acc:{test_acc: .4f}")

    if mute:
        print(f"# IGNORE: --- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f},  Final-TestAcc: {test_acc:.4f}--- , Early-stopping-TestF1ma: {best_val_test_f1ma:.4f},  Final-TestF1ma: {test_f1ma:.4f}")
    else:
        print(f"--- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f}, Final-TestAcc: {test_acc:.4f} --- , Early-stopping-TestF1ma: {best_val_test_f1ma:.4f},  Final-TestF1ma: {test_f1ma:.4f}")

    return test_acc, best_val_test_acc, test_f1ma, best_val_test_f1ma


def pretrain(model, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    train_loader, val_loader, test_loader, eval_train_loader = dataloaders

    epoch_iter = tqdm(range(max_epoch))

    if isinstance(train_loader, list) and len(train_loader) ==1:
        train_loader = [train_loader[0].to(device)]
        eval_train_loader = train_loader
    if isinstance(val_loader, list) and len(val_loader) == 1:
        val_loader = [val_loader[0].to(device)]
        test_loader = val_loader

    for epoch in epoch_iter:
        model.train()
        loss_list = []

        for subgraph in train_loader:
            subgraph = subgraph.to(device)
            loss, loss_dict = model(subgraph, subgraph.ndata["feat"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(loss_list)
        epoch_iter.set_description(f"# Epoch {epoch} | train_loss: {train_loss:.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)
        
        if epoch == (max_epoch//2):
            evaluete(model, (eval_train_loader, val_loader, test_loader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)
    return model

from CKA import CudaCKA

def get_cka_score_between_models(hidden_list1, hidden_list2, cka, device):
    cka_score_array = []
    for i in range(min(len(hidden_list1), len(hidden_list2))):
        cka_score_array.append([])
        for j in range(min(len(hidden_list1[i]), len(hidden_list2[i]))):
            cka_score = cka.linear_CKA(hidden_list1[i][j].to(device), hidden_list2[i][j].to(device)).cpu()
            cka_score_array[i].append(cka_score)

    return cka_score_array

def get_cka_score_between_layers(hidden_list, data_feats, cka, device):
    cka_score_array = []
    for ind, (hidden, data_feat) in enumerate(zip(hidden_list, data_feats)):
        cka_score_array.append([cka.linear_CKA(data_feat.to(device), hidden[0].to(device)).cpu()])
        for i in range(len(hidden) - 1):
            cka_score = cka.linear_CKA(hidden[i].to(device), hidden[i + 1].to(device)).cpu()
            cka_score_array[ind].append(cka_score)
    return cka_score_array

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

    (
        train_dataloader,
        valid_dataloader, 
        test_dataloader, 
        eval_train_dataloader, 
        num_features, 
        num_classes
    ) = load_inductive_dataset(dataset_name)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    f1ma_list = []
    estp_f1ma_list = []

    thin_acc_list = []
    thin_estp_acc_list = []
    thin_f1ma_list = []
    thin_estp_f1ma_list = []
    thinner_acc_list = []
    thinner_estp_acc_list = []
    thinner_f1ma_list = []
    thinner_estp_f1ma_list = []

    new_thin_flag = True

    if new_thin_flag:
        new_thin_acc_list = []
        new_thin_estp_acc_list = []
        new_thin_f1ma_list = []
        new_thin_estp_f1ma_list = []
        # thin_new_thin_acc_list = []
        # thin_new_thin_estp_acc_list = []
        # thin_new_thin_f1ma_list = []
        # thin_new_thin_estp_f1ma_list = []
        # thinner_new_thin_acc_list = []
        # thinner_new_thin_estp_acc_list = []
        # thinner_new_thin_f1ma_list = []
        # thinner_new_thin_estp_f1ma_list = []
    
    # wzb edit
    fat_tune_acc_list = []
    fat_tune_estp_acc_list = []
    thin_tune_acc_list = []
    thin_tune_estp_acc_list = []

    cka = CudaCKA(device=device)
    other_thin = True

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)

        # wzb edit
        # model_thin = build_thin_model(args)
        model_thin = build_model(args, if_thin=True, other_thin=other_thin, thin_base=2)
        model_thin = model_thin.to(device)
        model_thin_state_dict = model_thin.state_dict()

        model_thinner = build_model(args, if_thin=True, other_thin=other_thin, thin_base=4)
        model_thinner = model_thinner.to(device)
        model_thinner_state_dict = model_thinner.state_dict()
        
        # wzb edit
        if new_thin_flag:
            other_thin = True
            model_new_thin = build_model(args, if_thin=True, other_thin=other_thin, thin_base=1, thin_type='new_thin')
            model_new_thin = model_new_thin.to(device)
            model_new_thin_state_dict = model_new_thin.state_dict()

            # model_thin_new_thin = build_model(args, if_thin=True, other_thin=other_thin, thin_base=2, thin_type='new_thin')
            # model_thin_new_thin = model_thin_new_thin.to(device)
            # model_thin_new_thin_state_dict = model_thin_new_thin.state_dict()

            # model_thinner_new_thin = build_model(args, if_thin=True, other_thin=other_thin, thin_base=4, thin_type='new_thin')
            # model_thinner_new_thin = model_thinner_new_thin.to(device)
            # model_thinner_new_thin_state_dict = model_thinner_new_thin.state_dict()

        print('model encoder params', sum(p.numel() for p in model.encoder.parameters()))
        print('model thin encoder params', sum(p.numel() for p in model_thin.encoder.parameters()))
        print('model thinner encoder params', sum(p.numel() for p in model_thinner.encoder.parameters()))

        if new_thin_flag:
            print('model drop layers encoder params', sum(p.numel() for p in model_new_thin.encoder.parameters()))
            # print('model thin drop layers encoder params', sum(p.numel() for p in model_thin_new_thin.encoder.parameters()))
            # print('model thinner drop layers encoder params', sum(p.numel() for p in model_thinner_new_thin.encoder.parameters()))

        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        if not load_model:
            model = pretrain(model, (train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
        model = model.cpu()

        model = model.to(device)
        model.eval()

        if load_model:  # False
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:  # False
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        # wzb edit
        model_state_dict = model.state_dict()
        # model linear probe
        model = model.to(device)
        model.eval()

        model_thin = model_thin.to(device)
        model_thin.eval()
        model_thinner = model_thinner.to(device)
        model_thinner.eval()

        if new_thin_flag:
            model_new_thin = model_new_thin.to(device)
            model_new_thin.eval()
            # model_thin_new_thin = model_thin_new_thin.to(device)
            # model_thin_new_thin.eval()
            # model_thinner_new_thin = model_thinner_new_thin.to(device)
            # model_thinner_new_thin.eval()

        (final_acc, estp_acc, final_f1ma, estp_f1ma), hidden_list = evaluete(model, (eval_train_dataloader, valid_dataloader, test_dataloader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)
        f1ma_list.append(final_f1ma)
        estp_f1ma_list.append(estp_f1ma)

        # # wzb edit
        # # model fine tune
        # model = model.to(device)
        # model.train()
        # fat_tune_final_acc, fat_tune_estp_acc = evaluete(model, (eval_train_dataloader, valid_dataloader, test_dataloader), num_classes, 0.0002, weight_decay_f, max_epoch_f * 2, device, False)
        # fat_tune_acc_list.append(fat_tune_final_acc)
        # fat_tune_estp_acc_list.append(fat_tune_estp_acc)

        # # wzb edit
        # # thin fine tune
        # teacher_weights = model_state_dict
        # student_weights = model_thin_state_dict
        # model_thin.load_state_dict(get_slim_weight(teacher_weights, student_weights))
        # model_thin = model_thin.to(device)
        # model_thin.train()
        # thin_tune_final_acc, thin_tune_estp_acc = evaluete(model_thin, (eval_train_dataloader, valid_dataloader, test_dataloader), num_classes, 0.0002, weight_decay_f, max_epoch_f * 2, device, False)
        # thin_tune_acc_list.append(thin_tune_final_acc)
        # thin_tune_estp_acc_list.append(thin_tune_estp_acc)

        teacher_weights = model_state_dict
        student_weights = model_thin_state_dict
        weight_selection = get_slim_weight(teacher_weights, student_weights)
        model_thin.load_state_dict(weight_selection)

        teacher_weights = model_state_dict
        student_weights = model_thinner_state_dict
        weight_selection = get_slim_weight(teacher_weights, student_weights)
        model_thinner.load_state_dict(weight_selection)

        (thin_final_acc, thin_estp_acc, thin_final_f1ma, thin_estp_f1ma), thin_hidden_list = evaluete(model_thin, (eval_train_dataloader, valid_dataloader, test_dataloader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        thin_acc_list.append(thin_final_acc)
        thin_estp_acc_list.append(thin_estp_acc)
        thin_f1ma_list.append(thin_final_f1ma)
        thin_estp_f1ma_list.append(thin_estp_f1ma)

        (thinner_final_acc, thinner_estp_acc, thinner_final_f1ma, thinner_estp_f1ma), thinner_hidden_list = evaluete(model_thinner, (eval_train_dataloader, valid_dataloader, test_dataloader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        thinner_acc_list.append(thinner_final_acc)
        thinner_estp_acc_list.append(thinner_estp_acc)
        thinner_f1ma_list.append(thinner_final_f1ma)
        thinner_estp_f1ma_list.append(thinner_estp_f1ma)

        if new_thin_flag:
            teacher_weights = model_state_dict
            student_weights = model_new_thin_state_dict
            weight_selection = get_slim_weight(teacher_weights, student_weights)
            model_new_thin.load_state_dict(weight_selection)

            # teacher_weights = model_state_dict
            # student_weights = model_thin_new_thin_state_dict
            # weight_selection = get_slim_weight(teacher_weights, student_weights)
            # model_thin_new_thin.load_state_dict(weight_selection)

            # teacher_weights = model_state_dict
            # student_weights = model_thinner_new_thin_state_dict
            # weight_selection = get_slim_weight(teacher_weights, student_weights)
            # model_thinner_new_thin.load_state_dict(weight_selection)

            (new_thin_final_acc, new_thin_estp_acc, new_thin_final_f1ma, new_thin_estp_f1ma), thin_new_hidden_list = evaluete(model_new_thin, (eval_train_dataloader, valid_dataloader, test_dataloader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
            new_thin_acc_list.append(new_thin_final_acc)
            new_thin_estp_acc_list.append(new_thin_estp_acc)
            new_thin_f1ma_list.append(new_thin_final_f1ma)
            new_thin_estp_f1ma_list.append(new_thin_estp_f1ma)

            # thin_new_thin_final_acc, thin_new_thin_final_estp_acc = evaluete(model_thin_new_thin, (eval_train_dataloader, valid_dataloader, test_dataloader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
            # thin_new_thin_acc_list.append(thin_new_thin_final_acc)
            # thin_new_thin_estp_acc_list.append(thin_new_thin_final_estp_acc)

            # thinner_new_thin_final_acc, thinner_new_thin_final_estp_acc = evaluete(model_thinner_new_thin, (eval_train_dataloader, valid_dataloader, test_dataloader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
            # thinner_new_thin_acc_list.append(thinner_new_thin_final_acc)
            # thinner_new_thin_estp_acc_list.append(thinner_new_thin_final_estp_acc)

        if i == 0:
            for subgraph in eval_train_dataloader:
                subgraph = subgraph.to(device)
                train_data_feat = subgraph.ndata["feat"]
                break
            for subgraph in valid_dataloader:
                subgraph = subgraph.to(device)
                valid_data_feat = subgraph.ndata["feat"]
                break
            for subgraph in test_dataloader:
                subgraph = subgraph.to(device)
                test_data_feat = subgraph.ndata["feat"]
                break
            data_feats = [train_data_feat, valid_data_feat, test_data_feat]
            id2mode = {0: 'train', 1: 'valid', 2: 'test'}
            
            print('-' * 50 + '\n')
            cka_scores_between_model_and_thin = get_cka_score_between_models(hidden_list, thin_hidden_list, cka, device)        
            for i in range(len(cka_scores_between_model_and_thin)):
                for j in range(len(cka_scores_between_model_and_thin[i])):
                    print(f'cka_score_between_model_and_thin_layer_{j} in {id2mode[i]}: {cka_scores_between_model_and_thin[i][j]}')

            print('-' * 50 + '\n')
            cka_scores_between_model_and_thinner = get_cka_score_between_models(hidden_list, thinner_hidden_list, cka, device)
            for i in range(len(cka_scores_between_model_and_thinner)):
                for j in range(len(cka_scores_between_model_and_thinner[i])):
                    print(f'cka_score_between_model_and_thinner_layer_{j} in {id2mode[i]}: {cka_scores_between_model_and_thinner[i][j]}')
            
            print('-' * 50 + '\n')
            cka_scores_between_model_and_new_thin = get_cka_score_between_models(hidden_list, thin_new_hidden_list, cka, device)
            for i in range(len(cka_scores_between_model_and_new_thin)):
                for j in range(len(cka_scores_between_model_and_new_thin[i])):
                    print(f'cka_score_between_model_and_new_thin_layer_{j} in {id2mode[i]}: {cka_scores_between_model_and_new_thin[i][j]}')
            
            print('-' * 50 + '\n')
            cka_scores_between_model_layers = get_cka_score_between_layers(hidden_list, data_feats, cka, device)
            for i in range(len(cka_scores_between_model_layers)):
                for j in range(len(cka_scores_between_model_layers[i])):
                    print(f'cka_score_between_model_layer_{j-1}_and _layer{j} in {id2mode[i]}: {cka_scores_between_model_layers[i][j]}')
            
            print('-' * 50 + '\n')
            cka_scores_between_model_thin_layers = get_cka_score_between_layers(thin_hidden_list, data_feats, cka, device)
            for i in range(len(cka_scores_between_model_thin_layers)):
                for j in range(len(cka_scores_between_model_thin_layers[i])):
                    print(f'cka_score_between_model_thin_layer_{j-1}_and _layer{j} in {id2mode[i]}: {cka_scores_between_model_thin_layers[i][j]}')

            print('-' * 50 + '\n')
            cka_scores_between_model_thinner_layers = get_cka_score_between_layers(thinner_hidden_list, data_feats, cka, device)
            for i in range(len(cka_scores_between_model_thinner_layers)):
                for j in range(len(cka_scores_between_model_thinner_layers[i])):
                    print(f'cka_score_between_model_thinner_layer_{j-1}_and _layer{j} in {id2mode[i]}: {cka_scores_between_model_thinner_layers[i][j]}')

            print('-' * 50 + '\n')
            cka_scores_between_model_new_layers = get_cka_score_between_layers(thin_new_hidden_list, data_feats, cka, device)
            for i in range(len(cka_scores_between_model_new_layers)):
                for j in range(len(cka_scores_between_model_new_layers[i])):
                    print(f'cka_score_between_model_new_layer_{j-1}_and _layer{j} in {id2mode[i]}: {cka_scores_between_model_new_layers[i][j]}')
            
            print('-' * 50 + '\n')

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, es_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_f1mi: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_f1mi: {estp_acc:.4f}±{es_acc_std:.4f}")

    # wzb edit
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

    final_f1ma, final_f1ma_std = np.mean(f1ma_list), np.std(f1ma_list)
    estp_f1ma, es_f1ma_std = np.mean(estp_f1ma_list), np.std(estp_f1ma_list)
    print(f"# final_f1ma: {final_f1ma:.4f}±{final_f1ma_std:.4f}")
    print(f"# early-stopping_f1ma: {estp_f1ma:.4f}±{es_f1ma_std:.4f}")

    # # wzb edit
    # fat_tune_final_acc, fat_tune_final_acc_std = np.mean(fat_tune_acc_list), np.std(fat_tune_acc_list)
    # fat_tune_estp_acc, fat_tune_estp_acc_std = np.mean(fat_tune_estp_acc_list), np.std(fat_tune_estp_acc_list)
    # print(f"# fat_tune_final_acc: {fat_tune_final_acc:.4f}±{fat_tune_final_acc_std:.4f}")
    # print(f"# fat_tune_early-stopping_acc: {fat_tune_estp_acc:.4f}±{fat_tune_estp_acc_std:.4f}")
    # thin_tune_final_acc, thin_tune_final_acc_std = np.mean(thin_tune_acc_list), np.std(thin_tune_acc_list)
    # thin_tune_estp_acc, thin_tune_estp_acc_std = np.mean(thin_tune_estp_acc_list), np.std(thin_tune_estp_acc_list)
    # print(f"# thin_tune_final_acc: {thin_tune_final_acc:.4f}±{thin_tune_final_acc_std:.4f}")
    # print(f"# thin_tune_early-stopping_acc: {thin_tune_estp_acc:.4f}±{thin_tune_estp_acc_std:.4f}")

    if new_thin_flag:
        new_thin_final_acc, new_thin_final_acc_std = np.mean(new_thin_acc_list), np.std(new_thin_acc_list)
        new_thin_estp_acc, new_thin_estp_acc_std = np.mean(new_thin_estp_acc_list), np.std(new_thin_estp_acc_list)
        print(f"# new_thin_final_f1mi: {new_thin_final_acc:.4f}±{new_thin_final_acc_std:.4f}")
        print(f"# new_thin_early-stopping_f1mi: {new_thin_estp_acc:.4f}±{new_thin_estp_acc_std:.4f}")
        new_thin_final_f1ma, new_thin_final_f1ma_std = np.mean(new_thin_f1ma_list), np.std(new_thin_f1ma_list)
        new_thin_estp_f1ma, new_thin_estp_f1ma_std = np.mean(new_thin_estp_f1ma_list), np.std(new_thin_estp_f1ma_list)
        print(f"# new_thin_final_f1ma: {new_thin_final_f1ma:.4f}±{new_thin_final_f1ma_std:.4f}")
        print(f"# new_thin_early-stopping_f1ma: {new_thin_estp_f1ma:.4f}±{new_thin_estp_f1ma_std:.4f}")

        # thin_new_thin_final_acc, thin_new_thin_final_acc_std = np.mean(thin_new_thin_acc_list), np.std(thin_new_thin_acc_list)
        # thin_new_thin_estp_acc, thin_new_thin_estp_acc_std = np.mean(thin_new_thin_estp_acc_list), np.std(thin_new_thin_estp_acc_list)
        # print(f"# thin_new_thin_final_acc: {thin_new_thin_final_acc:.4f}±{thin_new_thin_final_acc_std:.4f}")
        # print(f"# thin_new_thin_early-stopping_acc: {thin_new_thin_estp_acc:.4f}±{thin_new_thin_estp_acc_std:.4f}")

        # thinner_new_thin_final_acc, thinner_new_thin_final_acc_std = np.mean(thinner_new_thin_acc_list), np.std(thinner_new_thin_acc_list)
        # thinner_new_thin_estp_acc, thinner_new_thin_estp_acc_std = np.mean(thinner_new_thin_estp_acc_list), np.std(thinner_new_thin_estp_acc_list)
        # print(f"# thinner_new_thin_final_acc: {thinner_new_thin_final_acc:.4f}±{thinner_new_thin_final_acc_std:.4f}")
        # print(f"# thinner_new_thin_early-stopping_acc: {thinner_new_thin_estp_acc:.4f}±{thinner_new_thin_estp_acc_std:.4f}")

def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    return args


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
