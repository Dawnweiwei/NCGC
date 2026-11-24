# NCGC

This is the official code for the WSDM 2027 paper **Rethinking Semi-Supervised Node Classification with Self-Supervised Graph Clustering**.

## Requirements

We tested the training code with Python 3.10, PyTorch 1.12.1, and PyTorch Geometric 2.6.1. After installing PyTorch and PyTorch Geometric, install the remaining dependencies:

```bash
python -m pip install numpy==1.26.4 scipy==1.15.3 scikit-learn==1.6.1 ogb==1.3.6 gdown==4.7.3
```

## Dataset

All datasets will be automatically downloaded when running the corresponding experiments.  
The sources of each dataset are listed below:

- **Cora**, **Citeseer**, **Pubmed**: downloaded from `torch_geometric.datasets`
- **Amazon-Computer**, **Amazon-Photo**: downloaded from `torch_geometric.datasets`
- **Coauthor-CS**, **Coauthor-Physics**: downloaded from `torch_geometric.datasets`

---

## Run NCGC with GNNs on Citation Graphs and Medium Graphs

To run the 28 main NCGC configurations (four GNN backbones on seven datasets), use:

```bash
sh run.sh
```

Each command uses one node split across its training runs. The backbone is reinitialized for each run, while the DEC cluster centroids carry over between runs within the same command.

To run NCGC with GCNII on a single citation network, choose the corresponding command:

```bash
# Cora
python new_main.py --gnn gcnii --dataset cora --lr 0.001 --rand_split_class --local_layers 32 --hidden_channels 256 --dropout 0.6 --w1 0.85 --w2 0.15 --w3 1 --epochs 500 --runs 3 --beta 0.003

# Citeseer
python new_main.py --gnn gcnii --dataset citeseer --lr 0.001 --rand_split_class --local_layers 32 --hidden_channels 512 --dropout 0.7 --w1 0.6 --w2 0.4 --w3 1 --epochs 500 --beta 0.003

# Pubmed
python new_main.py --gnn gcnii --dataset pubmed --lr 0.005 --rand_split_class --local_layers 32 --hidden_channels 256 --dropout 0.6 --w1 0.8 --w2 0.2 --w3 1 --epochs 500 --beta 0.004
```
