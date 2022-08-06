import argparse
import torch

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).') # should use 5e-6 for our method
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--model', type=str, default="SGC",
                        choices=["SGC", "GCN"],
                        help='model to use.')
    parser.add_argument('--feature', type=str, default="non",
                        choices=['non', 'mul', 'cat', 'adj'],
                        help='feature-type')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                       choices=['AugNormAdj'],
                       help='Normalization method for the adjacency matrix.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    parser.add_argument('--per', type=int, default=-1,
                        help='Number of each nodes so as to balance.')
    parser.add_argument('--experiment', type=str, default="base-experiment",
                        help='feature-type')
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')
    parser.add_argument('--strategy', type=str, default='random', help='query strategy')
    parser.add_argument('--file_io', type=int, default=0,
                        help='determine whether use file io')
    parser.add_argument('--reweight', type=int, default=1,
                        choices=[0, 1],
                        help='whether to use reweighting')
    parser.add_argument('--adaptive', type=int, default=1,
                        choices=[0, 1],
                        help='to use adaptive weighting')
    parser.add_argument('--lambdaa', type=float, default=0.99,
                        help='control combination')


    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
