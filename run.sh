
python new_main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --epochs 500 --lr 0.001 --runs 3 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 0 --ln --rand_split_class --beta 0.004
python new_main.py --gnn sage --dataset amazon-computer --hidden_channels 128 --epochs 1000 --lr 0.001 --runs 3 --local_layers 4 --weight_decay 5e-5 --dropout 0.3 --device 0 --ln --rand_split_class --w1 0.8 --w2 0.2 --w3 1 --beta 0.005
python new_main.py --gnn gat --dataset amazon-computer --hidden_channels 256 --epochs 500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-5 --dropout 0.7 --device 0 --ln --rand_split_class --w1 0.8 --w2 0.2 --w3 1 --bn --iterations 1 --epsilon 0.03 --beta 0.005

python new_main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --epochs 500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-5 --dropout 0.6 --device 0 --ln --res --rand_split_class --w1 0.9 --w2 0.1 --w3 1 --beta 0.007
python new_main.py --gnn sage --dataset amazon-photo --hidden_channels 64 --epochs 500 --lr 0.001 --runs 3 --local_layers 6 --weight_decay 5e-5 --dropout 0.2 --device 0 --ln --res --rand_split_class --w1 0.9 --w2 0.1 --w3 1 --beta 0.005
python new_main.py --gnn gat --dataset amazon-photo --hidden_channels 64 --epochs 500 --lr 0.001 --runs 3 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 0 --ln --res --rand_split_class --w1 0.5 --w2 0.5 --w3 1 --beta 0.003

python new_main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --epochs 500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 0 --ln --res --rand_split_class --w1 0.3 --w2 0.7 --w3 1 --bn --beta 0.003
python new_main.py --gnn sage --dataset coauthor-cs --hidden_channels 512 --epochs 500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.5 --device 0 --ln --res --rand_split_class --w1 0.8 --w2 0.2 --w3 1 --beta 0.001
python new_main.py --gnn gat --dataset coauthor-cs --hidden_channels 256 --epochs 500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.5 --device 0 --ln --res --rand_split_class --w1 0.9 --w2 0.1 --w3 1 --beta 0.004 --iterations 3 --epsilon 0.04

python new_main.py --gnn gcn --dataset coauthor-physics --hidden_channels 64 --epochs 500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.5 --device 0 --rand_split_class --ln --w1 0.4 --w2 0.6 --w3 1 --beta 0.008
python new_main.py --gnn sage --dataset coauthor-physics --hidden_channels 64 --epochs 500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --device 0 --rand_split_class --w1 0.8 --w2 0.2 --w3 1 --beta 0.003
python new_main.py --gnn gat --dataset coauthor-physics --hidden_channels 128 --epochs 500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.6 --device 0  --res --rand_split_class --w1 0.3 --w2 0.7 --w3 1 --beta 0.004


python new_main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 512 --weight_decay 5e-4 --dropout 0.8 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 5 --w1 0.25 --w2 0.75 --w3 1 --pre_linear --beta 0.003
python new_main.py --gnn sage --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 5 --w1 1 --w2 1 --w3 1 --pre_linear --w1 1 --w2 1 --w3 1 --beta 0.005
python new_main.py --gnn gat --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 5 --res --w1 0.6 --w2 0.4 --w3 1 --epsilon 0.04 --beta 0.003

python new_main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 5 --w1 0.85 --w2 0.15 --w3 1 --epsilon 0.03 --beta 0.008
python new_main.py --gnn sage --dataset citeseer --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 0.01 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 5 --w1 0.9 --w2 0.1 --w3 1 --beta 0.005
python new_main.py --gnn gat --gnn gat --dataset citeseer --lr 0.001 --local_layers 3 --hidden_channels 256 --weight_decay 0.01 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 5 --res --w1 0.8 --w2 0.2 --w3 1 --beta 0.002

python new_main.py --gnn gcn --dataset pubmed --lr 0.001 --local_layers 2 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 5 --w1 0.90 --w2 0.10 --w3 1 --epsilon 0.04 --iterations 4 --beta 0.008
python new_main.py --gnn sage --dataset pubmed --lr 0.001 --local_layers 4 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 5 --w1 0.25 --w2 0.75 --w3 1 --beta 0.004
python new_main.py --gnn gat --dataset pubmed --lr 0.01 --local_layers 2 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 5 --w1 0.90 --w2 0.10 --w3 1 --beta 0.008


python new_main.py --gnn gcnii --dataset cora --lr 0.001 --rand_split_class --local_layers 32 --hidden_channels 256 --dropout 0.6 --w1 0.85 --w2 0.15 --w3 1 --epochs 500 --runs 3 --beta 0.003
python new_main.py --gnn gcnii --dataset citeseer --lr 0.001 --rand_split_class --local_layers 32 --hidden_channels 512 --dropout 0.7 --w1 0.6 --w2 0.4 --w3 1 --epochs 500 --beta 0.003
python new_main.py --gnn gcnii --dataset pubmed --lr 0.005 --rand_split_class --local_layers 32 --hidden_channels 256 --dropout 0.6 --w1 0.8 --w2 0.2 --w3 1 --epochs 500 --beta 0.004
python new_main.py --gnn gcnii --dataset coauthor-cs --lr 0.005 --rand_split_class --local_layers 32 --hidden_channels 256 --dropout 0.6 --w1 0.9 --w2 0.1 --w3 1 --epochs 1000 --beta 0.002
python new_main.py --gnn gcnii --dataset coauthor-physics --lr 0.005 --rand_split_class --local_layers 32 --hidden_channels 256 --dropout 0.5 --w1 0.8 --w2 0.2 --w3 1 --epochs 1000 --beta 0.005
python new_main.py --gnn gcnii --dataset amazon-computer --hidden_channels 256 --lr 0.01 --runs 3 --local_layers 2 --weight_decay 5e-5 --dropout 0.6 --device 0 --ln --rand_split_class --epochs 500 --w1 0.8 --w2 0.2 --w3 1 --beta 0.003
python new_main.py --gnn gcnii --dataset amazon-photo --lr 0.01 --rand_split_class --local_layers 2 --hidden_channels 512 --dropout 0.6 --w1 0.8 --w2 0.2 --w3 1 --epochs 500 --bn --beta 0.008


