Source code of our NeurIPS 2024paper "Uncovering the Redundancy in Graph Self-supervised Learning Models".
# Usage
```bash
# GRACE
cd GRACE
python train.py --dataset Cora
```
The `--dataset` argument should be one of [ Cora, CiteSeer, PubMed, Computers, Photo ].

```bash
# GraphMAE
cd GraphMAE
sh scripts/run_transductive.sh <dataset_name> <gpu_id>
```
The `<dataset_name>` agument should be one of [ cora, citeseer, pubmed, computer, photo, ogbn-arxiv ].

```bash
# MaskGAE
# Cora
python train_nodeclas.py --dataset Cora --bn --l2_normalize --alpha 0.004 --full_data --save_path Cora_test_new_thin.pt
# Citeseer
python train_nodeclas.py --dataset Citeseer --bn --l2_normalize --nodeclas_weight_decay 0.1 --alpha 0.001 --lr 0.02 --full_data --save_path Citeseer_test_new_thin.pt
# Pubmed
python train_nodeclas.py --dataset Pubmed --bn --l2_normalize --alpha 0.001  --encoder_dropout 0.5 --decoder_dropout 0.5 --full_data --save_path PubMed_test_new_thin.pt
# Photo
python train_nodeclas.py --dataset Photo --bn --nodeclas_weight_decay 5e-3 --decoder_channels 128 --lr 0.005 --save_path Photo_test_new_thin.pt
# Computers
python train_nodeclas.py --dataset Computers --bn --encoder_dropout 0.5 --alpha 0.002 --encoder_channels 128 --hidden_channels 256 --eval_period 20 --save_path Computers_test_new_thin.pt
# arxiv
python train_nodeclas.py --dataset arxiv --bn --decoder_channels 128 --decoder_dropout 0. --decoder_layers 4 --encoder_channels 256 --encoder_dropout 0.2 --encoder_layers 4 --hidden_channels 512 --lr 0.0005 --nodeclas_weight_decay 0 --weight_decay 0.0001 --epochs 100   --eval_period 10 --save_path arxiv_test_new_thin.pt
```

