from time import perf_counter
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_reddit_data, sgc_precompute, set_seed
from metrics import f1
from models import SGC
import os

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--inductive', action='store_true', default=False,
                    help='inductive training.')
parser.add_argument('--test', action='store_true', default=False,
                    help='inductive training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                   choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN',
                            'AugNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm'],
                   help='Normalization method for the adjacency matrix.')
parser.add_argument('--model', type=str, default="SGC",
                    help='model to use.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--optimizer', type=str, default='LBFGS',
                    help='optimizer to use')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

set_seed(args.seed, args.cuda)

adj, train_adj, features, labels, idx_train, idx_val, idx_test = load_reddit_data(args.normalization, cuda=args.cuda)
print("Finished data loading.")

model = SGC(features.size(1), labels.max().item()+1)

print("# Feature: {}, # Classes: {}".format(features.size(1), labels.max().item()+1))

if args.cuda: model.cuda()

processed_features, precompute_time = sgc_precompute(features, adj, args.degree)

if args.inductive:
    train_features, _ = sgc_precompute(features[idx_train], train_adj, args.degree)
else:
    train_features = processed_features[idx_train]

test_features = processed_features[idx_test if args.test else idx_val]

def train_regression(model, train_features, train_labels, epochs, optimizer='Adam'):
    # optimizer = optim.LBFGS(model.parameters(), lr=1)
    if optimizer != 'Adam':
        optimizer = optim.LBFGS(model.parameters(), lr=1)
        def closure():
            optimizer.zero_grad()
            output = model(train_features)
            loss_train = F.cross_entropy(output, train_labels)
            loss_train.backward()
            return loss_train

        t = perf_counter()
        for epoch in range(epochs):
            loss_train = optimizer.step(closure)

        train_time = perf_counter() - t
        return model, train_time

    optimizer = optim.Adam(model.parameters(), lr=0.1,
                               weight_decay=5e-6)
    model.train()
    forward_time = 0
    cross_entropy_time = 0
    backward_time = 0
    step_time = 0
    softmax_time = 0
    nll_time = 0
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # forward time
        t_forward = perf_counter()
        output = model(train_features)
        forward_time += perf_counter() - t_forward

        # Cross Entropy time
        t_CE = perf_counter()
        # loss_train = F.cross_entropy(output, train_labels)

        t_softmax_log = perf_counter()
        softmax_log = F.log_softmax(output,dim=1)
        softmax_time += perf_counter() - t_softmax_log

        t_nll = perf_counter()
        loss_train = F.nll_loss(softmax_log, train_labels)
        nll_time += perf_counter() - t_nll

        cross_entropy_time += perf_counter() - t_CE

        # Backward time
        t_backward = perf_counter()
        loss_train.backward()
        backward_time += perf_counter() - t_backward

        # Step time
        t_step = perf_counter()
        optimizer.step()
        step_time += perf_counter() - t_step

    train_time = perf_counter()-t
    return model, train_time, forward_time, cross_entropy_time, backward_time, step_time, softmax_time, nll_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return f1(model(test_features), test_labels)

def print_time_ratio(name, time1, train_time):
    print("{}: {:.4f}s, ratio: {}".format(name, time1, time1/train_time))

def save_time_result(file_name, *args):
    # args is the names of the time
    save_dict = {}
    save_list = []
    for arg in args:
        save_list.append(arg)

    for x in save_list:
        save_dict[x] = eval(x)
    # print(save_dict)
    import pickle
    with open(file_name, 'wb') as f:
        pickle.dump(save_dict, f)


if args.optimizer == 'Adam':
    model, train_time, forward_time, cross_entropy_time, backward_time, step_time, \
                softmax_time, nll_time = train_regression(model, train_features, labels[idx_train], args.epochs, optimizer=args.optimizer)

else:
    model, train_time = train_regression(model, train_features, labels[idx_train], args.epochs, optimizer=args.optimizer)


test_f1, _ = test_regression(model, test_features, labels[idx_test if args.test else idx_val])
total_time = train_time + precompute_time
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time,
                                                                                  total_time))
print("Total Time: {:.4f}s, {} F1: {:.4f}".format(train_time+precompute_time,
                                                    "Test" if args.test else "Val",
                                                    test_f1))

if args.optimizer == 'Adam':
    print_time_ratio('Forward Time', forward_time, train_time)
    print_time_ratio('Cross Entropy Time', cross_entropy_time, train_time)
    print("--Cross Entropy Time Details--")
    print_time_ratio('Softmax_log Time', softmax_time, train_time)
    print_time_ratio('NLL Time', nll_time, train_time)
    print_time_ratio('Backward Time', backward_time, train_time)
    print_time_ratio('Step Time', step_time, train_time)

    file_name = os.path.join('time_result', 'reddit')
    save_time_result(file_name, 'total_time', 'precompute_time', 'train_time', 'forward_time', 'cross_entropy_time',
                     'softmax_time',
                     'nll_time', 'backward_time', 'step_time')
