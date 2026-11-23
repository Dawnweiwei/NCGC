## Dataset

All datasets will be automatically downloaded when running the corresponding experiments.  
The sources of each dataset are listed below:

- **Cora**, **Citeseer**, **Pubmed**: downloaded from `torch_geometric.datasets`
- **Amazon-Computer**, **Amazon-Photo**: downloaded from `torch_geometric.datasets`
- **Coauthor-CS**, **Coauthor-Physics**: downloaded from `torch_geometric.datasets`

---

## Run NCGC with GNNs on Citation Graphs and Medium Graphs

**GNNs**:  
To reproduce all NCGC experiments, run:
```bash
sh run.sh

You can also execute a single command to run a specific experiment.
For example, to run NCGC (GCNII) on Planetoid datasets:

# Cora
python new_main.py --gnn gcnii --dataset cora --lr 0.001 --rand_split_class --local_layers 32 --hidden_channels 256 --dropout 0.6 --w1 0.85 --w2 0.15 --w3 1 --epochs 500 --runs 3 --beta 0.003

# Citeseer
python new_main.py --gnn gcnii --dataset citeseer --lr 0.001 --rand_split_class --local_layers 32 --hidden_channels 512 --dropout 0.7 --w1 0.6 --w2 0.4 --w3 1 --epochs 500 --beta 0.003

# Pubmed
python new_main.py --gnn gcnii --dataset pubmed --lr 0.005 --rand_split_class --local_layers 32 --hidden_channels 256 --dropout 0.6 --w1 0.8 --w2 0.2 --w3 1 --epochs 500 --beta 0.004
