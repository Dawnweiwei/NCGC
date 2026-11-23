import math
import argparse
import random
import numpy as np
import torch.nn as nn
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits
from eval import *
from parse import parse_method, parser_add_main_args

from Soft_Assignment import SoftClusterAssignment


def fix_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def sinkhorn(Q, epsilon=0.04, sinkhorn_iterations=3):
    """Balance class and node masses in log space, then return per-node targets."""
    num_nodes, num_classes = Q.shape
    log_num_nodes = math.log(num_nodes)
    log_num_classes = math.log(num_classes)
    scores = Q / epsilon
    log_mass = torch.log_softmax(scores.flatten(), dim=0).reshape_as(scores)

    for _ in range(sinkhorn_iterations):
        log_mass = torch.log_softmax(log_mass, dim=0) - log_num_classes
        log_mass = torch.log_softmax(log_mass, dim=1) - log_num_nodes

    return (log_mass + log_num_nodes).exp()


parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num, args.dataset)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

dataset.label = dataset.label.to(device)

print("train idx example:", split_idx_lst[0]['train'][:10])
print("valid idx example:", split_idx_lst[0]['valid'][:10])
print("test idx example:", split_idx_lst[0]['test'][:10])

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)


### Load method ###
model = parse_method(args, n, c, d, device)
soft_assign = SoftClusterAssignment(
    num_cluster=c,
    hidden_dim=model.hidden_channels,
    alpha=1.0
).to(device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('questions'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

args.method = args.gnn
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    if args.dataset in (
    'coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo', 'cora', 'citeseer', 'pubmed','wiki-cs'):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[0]
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)
    train_idx = train_idx.long()


    model.reset_parameters()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(soft_assign.parameters()),
                                 weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_test = float('-inf')
    if args.save_model:
        save_model(args, model, optimizer, run)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        all_idx = torch.arange(dataset.graph['node_feat'].size(0), device=device)
        non_train_idx = all_idx[~torch.isin(all_idx, train_idx)]


        out, h = model(dataset.graph['node_feat'], dataset.graph['edge_index'])  # h as hidden
        # prob = F.softmax(out, dim=1)
        out = F.log_softmax(out, dim=1)


        loss_labeled = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])


        all_idx = torch.arange(out.shape[0], device=device)
        non_train_idx = all_idx[~torch.isin(all_idx, train_idx)]

        if len(non_train_idx) > 0:

            # # without sinkhorn
            # soft_target = torch.nn.functional.softmax(out[non_train_idx].detach(), dim=1)
            # log_probs = torch.nn.functional.log_softmax(out[non_train_idx], dim=1)
            # sinkhorn_loss = -(soft_target * log_probs).sum(dim=1).mean()


            Q = sinkhorn(out[non_train_idx],epsilon=args.epsilon,sinkhorn_iterations=args.iterations) # Q.shape = [batch_size, num_classes]
            # log_softmax
            log_probs = torch.nn.functional.log_softmax(out[non_train_idx], dim=1)

            sinkhorn_loss = -(Q.detach() * log_probs).sum(dim=1).mean()


        q = soft_assign(h)
        p = soft_assign._get_target_distribution(q.detach())
        KL_div = F.kl_div(q.log(), p, reduction='batchmean')

        loss = args.w1 * loss_labeled + args.w2 * sinkhorn_loss + args.w3 * KL_div
        # loss = loss_labeled


        loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]
            if args.save_model:
                save_model(args, model, optimizer, run)

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'KL: {KL_div.item():.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * best_test:.2f}%')

    logger.print_statistics(run)

results = logger.print_statistics()
### Save results ###
save_result(args, results)