import torch
from model import MPNNs



def parse_method(args, n, c, d, device):
    model = MPNNs(d, args.hidden_channels, c,
                  beta=args.beta,
                  local_layers=args.local_layers,
                  dropout=args.dropout,
                  heads=args.num_heads,
                  pre_ln=args.pre_ln,
                  pre_linear=args.pre_linear,
                  res=args.res,
                  ln=args.ln,
                  bn=args.bn,
                  jk=args.jk,
                  gnn=args.gnn).to(device)


    return model




def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=3,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')

    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')

    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    parser.add_argument('--model', type=str, default='MPNN')
    # GNN
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
    parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--jk', action='store_true', help='use JK for GNNs')

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')
    parser.add_argument('--w1', type=float, default=0.8, help='labeled loss weight')
    parser.add_argument('--w2', type=float, default=0.2, help='sinkhorn loss weight')
    parser.add_argument('--w3', type=float, default=1.0, help='KL divergence weight')
    parser.add_argument('--beta', type=float, default=0.005, help='soft orthogonal weight')
    parser.add_argument("--epsilon", type=float, default=0.04, help="Sinkhorn epsilon value")
    parser.add_argument("--iterations", type=int, default=3, help="Sinkhorn iterations")
