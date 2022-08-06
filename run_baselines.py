import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation_multi_steps, early_stopping, remove_nodes_from_walks, sgc_precompute, \
    get_classes_statistic, load_reddit_data, load_Amazon, load_coauthor
from models import get_model
from metrics import accuracy, f1
import pickle as pkl
from args import get_citation_args
from time import perf_counter
from sampling_methods import *
import os
import datetime
import json

# Arguments
args = get_citation_args()


def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    best_acc_val = 0
    should_stop = False
    stopping_step = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                output = model(val_features)
                acc_val = accuracy(output, val_labels)
                best_acc_val, stopping_step, should_stop = early_stopping(acc_val, best_acc_val, stopping_step,
                                                                          flag_step=10)
                if acc_val == best_acc_val:
                    # save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, 'checkpoint_SGC.pt')
                if should_stop:
                    print('epoch: {}, acc_val: {}, best_acc_val: {}'.format(epoch, acc_val, best_acc_val))
                    # load best model
                    checkpoint = torch.load('checkpoint_SGC.pt')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    break

    train_time = perf_counter() - t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)
        micro_val, macro_val = f1(output, val_labels)
        print('acc_val: {}'.format(acc_val))
    return model, acc_val, micro_val, macro_val, train_time


def test_regression(model, test_features, test_labels):
    model.eval()
    output = model(test_features)
    acc_test = accuracy(output, test_labels)
    micro_test, macro_test = f1(output, test_labels)
    return acc_test, micro_test, macro_test

def train_GCN(model, adj, selected_nodes, val_nodes,
             features, train_labels, val_labels,
             epochs=args.epochs, weight_decay=args.weight_decay,
             lr=args.lr, dropout=args.dropout):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    best_acc_val = 0
    should_stop = False
    stopping_step = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        output = output[selected_nodes, :]
        # print(f'output.size(): {output.size()}')
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                output = model(features, adj)
                output = output[val_nodes, :]
                acc_val = accuracy(output, val_labels)
                best_acc_val, stopping_step, should_stop = early_stopping(acc_val, best_acc_val, stopping_step,
                                                                          flag_step=10)
                if acc_val == best_acc_val:
                    # save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f'checkpoint_{args.strategy}_{args.dataset}.pt')
                if should_stop:
                    print('epoch: {}, acc_val: {}, best_acc_val: {}'.format(epoch, acc_val, best_acc_val))
                    # load best model
                    checkpoint = torch.load(f'checkpoint_{args.strategy}_{args.dataset}.pt')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    break

    train_time = perf_counter() - t

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        output = output[val_nodes, :]
        acc_val = accuracy(output, val_labels)
        micro_val, macro_val = f1(output, val_labels)
        print('acc_val: {}'.format(acc_val))
    return model, acc_val, micro_val, macro_val, train_time

def test_GCN(model, adj, test_mask, features, test_labels):
    model.eval()
    output = model(features, adj)
    output = output[test_mask, :]
    acc_test = accuracy(output, test_labels)
    micro_test, macro_test = f1(output, test_labels)
    return acc_test, micro_test, macro_test


def print_time_ratio(name, time1, train_time):
    print("{}: {:.4f}s, ratio: {}".format(name, time1, time1 / train_time))


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


def ensure_nonrepeat(idx_train, selected_nodes):
    for node in idx_train:
        if node in selected_nodes:
            raise Exception(
                'In this iteration, the node {} need to be labelled is already in selected_nodes'.format(node))
    return


class run_wrapper():
    def __init__(self, dataset, normalization, cuda):
        if dataset in ['CS', 'Physics']:
            self.adj, self.graph, self.features, self.labels, self.idx_test, self.idx_non_test = load_coauthor(dataset,
                                                                                                   normalization,
                                                                                                   cuda=cuda)
        elif dataset != 'reddit':
            self.adj, self.graph, self.features, self.labels, self.idx_test, self.idx_non_test = load_citation_multi_steps(
                dataset, normalization, cuda=cuda)
            self.nx_G = nx.from_dict_of_lists(self.graph)
        else:
            self.adj, self.graph, self.train_adj, self.features, self.labels, self.idx_test, self.idx_non_test = load_reddit_data(
                normalization, cuda=cuda)
            self.nx_G = self.graph
        self.dataset = dataset
        print(f'self.labels: {self.labels}')
        print('finished loading dataset')
        self.raw_features = self.features
        if args.model == "SGC":
            self.features, precompute_time = sgc_precompute(self.features, self.adj, args.degree)
            print("{:.4f}s".format(precompute_time))
            if args.strategy == 'featprop':
                self.dis_features = self.features
        else:
            if args.strategy == 'featprop':
                self.dis_features, precompute_time = sgc_precompute(self.features, self.adj, args.degree)
                # torch.save(self.dis_features.data, 'visualization/featprop_feat.pt')
                # input('wait')


    def run(self, strategy, num_labeled_list=[10, 15, 20, 25, 30, 35, 40, 50], max_budget=160, seed=1):
        set_seed(seed, args.cuda)
        max_budget = num_labeled_list[-1]
        if strategy in ['ppr', 'pagerank', 'pr_ppr', 'mixed', 'mixed_random', 'unified']:
            print('strategy is ppr or pagerank')
            # nx_G = nx.from_dict_of_lists(self.graph)
            nx_G = self.nx_G
            PR_scores = nx.pagerank(nx_G, alpha=0.85)
            # print('PR_scores: ', PR_scores)
            nx_nodes = nx.nodes(nx_G)
            original_weights = {}
            for node in nx_nodes:
                original_weights[node] = 0.

        idx_non_test = self.idx_non_test.copy()
        print('len(idx_non_test) is {}'.format(len(idx_non_test)))
        # Select validation nodes.
        num_val = 500
        idx_val = np.random.choice(idx_non_test, num_val, replace=False)
        idx_non_test = list(set(idx_non_test) - set(idx_val))

        # initially select some nodes.
        L = 5
        selected_nodes = np.random.choice(idx_non_test, L, replace=False)
        idx_non_test = list(set(idx_non_test) - set(selected_nodes))

        model = get_model(args.model, self.features.size(1), self.labels.max().item() + 1, args.hidden, args.dropout,
                          args.cuda)

        budget = 20
        steps = 6
        pool = idx_non_test
        print('len(idx_non_test): {}'.format(len(idx_non_test)))
        np.random.seed() # cancel the fixed seed
        if args.model == 'GCN':
            args.lr = 0.01
            model, acc_val, micro_val, macro_val, train_time = train_GCN(model, self.adj, selected_nodes, idx_val, self.features,
                                                                                self.labels[selected_nodes],
                                                                                self.labels[idx_val],
                                                                                args.epochs, args.weight_decay, args.lr,
                                                                                args.dropout)
        else:
            model, acc_val, micro_val, macro_val, train_time = train_regression(model, self.features[selected_nodes],
                                                                                self.labels[selected_nodes],
                                                                                self.features[idx_val],
                                                                                self.labels[idx_val],
                                                                                args.epochs, args.weight_decay, args.lr,
                                                                                args.dropout)
        print('-------------initial results------------')
        print('micro_val: {:.4f}, macro_val: {:.4f}'.format(micro_val, macro_val))
        # Active learning
        print('strategy: ', strategy)
        cur_num = 0
        val_results = {'acc': [], 'micro': [], 'macro': []}
        test_results = {'acc': [], 'micro': [], 'macro': []}

        uncertainty_results = {}
        if strategy == 'rw':
            self.walks = remove_nodes_from_walks(self.walks, selected_nodes)
        if strategy == 'unified':
            nodes = nx.nodes(nx_G)
            uncertainty_score = get_uncertainty_score(model, self.features, nodes)
            init_weights = {n: float(uncertainty_score[n]) for n in nodes}
            for node in selected_nodes:
                init_weights[node] = 0
            uncertainty_results[5] = {'selected_nodes': selected_nodes.tolist(), 'uncertainty_scores': init_weights}


        time_AL = 0
        for i in range(len(num_labeled_list)):
            if num_labeled_list[i] > max_budget:
                break
            budget = num_labeled_list[i] - cur_num
            cur_num = num_labeled_list[i]
            t1 = perf_counter()
            if strategy == 'random':
                idx_train = query_random(budget, pool)
            elif strategy == 'uncertainty':
                if args.model == 'GCN':
                    idx_train = query_uncertainty_GCN(model, self.adj, self.features, budget, pool)
                else:
                    idx_train = query_uncertainty(model, self.features, budget, pool)
            elif strategy == 'largest_degrees':
                if args.dataset not in ['cora', 'citeseer', 'pubmed']:
                    idx_train = query_largest_degree(self.graph, budget, pool)
                else:
                    idx_train = query_largest_degree(nx.from_dict_of_lists(self.graph), budget, pool)
            elif strategy == 'coreset_greedy':
                idx_train = qeury_coreset_greedy(self.features, list(selected_nodes), budget, pool)
            elif strategy == 'featprop':
                idx_train = query_featprop(self.dis_features, budget, pool)
            elif strategy == 'pagerank':
                idx_train = query_pr(PR_scores, budget, pool)
            else:
                raise NotImplementedError('cannot find the strategy {}'.format(strategy))

            time_AL += perf_counter() - t1
            assert len(idx_train) == budget
            ensure_nonrepeat(idx_train, selected_nodes)
            selected_nodes = np.append(selected_nodes, idx_train)
            pool = list(set(pool) - set(idx_train))
            if args.model == 'GCN':
                model, acc_val, micro_val, macro_val, train_time = train_GCN(model, self.adj, selected_nodes, idx_val, self.features,
                                                                             self.labels[selected_nodes],
                                                                             self.labels[idx_val],
                                                                             args.epochs, args.weight_decay, args.lr,
                                                                             args.dropout)
            else:
                model, acc_val, micro_val, macro_val, train_time = train_regression(model, self.features[selected_nodes],
                                                                                    self.labels[selected_nodes],
                                                                                    self.features[idx_val],
                                                                                    self.labels[idx_val],
                                                                                    args.epochs, args.weight_decay, args.lr,
                                                                                    args.dropout)

            if args.model == 'GCN':
                acc_test, micro_test, macro_test = test_GCN(model, self.adj, self.idx_test, self.features,
                                                                   self.labels[self.idx_test])
            else:
                acc_test, micro_test, macro_test = test_regression(model, self.features[self.idx_test],
                                                                   self.labels[self.idx_test])

            acc_val = acc_val.cpu().item()
            acc_test = acc_test.cpu().item()

            acc_val = round(acc_val, 4)
            acc_test = round(acc_test, 4)
            micro_val = round(micro_val, 4)
            micro_test = round(micro_test, 4)
            macro_val = round(macro_val, 4)
            macro_test = round(macro_test, 4)

            val_results['acc'].append(acc_val)
            val_results['micro'].append(micro_val)
            val_results['macro'].append(macro_val)
            test_results['acc'].append(acc_test)
            test_results['micro'].append(micro_test)
            test_results['macro'].append(macro_test)
            print('micro_val: {:.4f}, macro_val: {:.4f}'.format(micro_val, macro_val))
            print('micro_test: {:.4f}, macro_test: {:.4f}'.format(micro_test, macro_test))

        print('AL Time: {}s'.format(time_AL))
        return val_results, test_results, get_classes_statistic(self.labels[selected_nodes].cpu().numpy()), time_AL


def print_avg_results(val_avg_results, test_avg_results):
    print('-------Average Results-------------')
    for metric in ['micro', 'macro']:
        print("Test_{}_f1 {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_avg_results[metric])))
        # print("Test_{}_std {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_std_results[metric])))
        print("Val_{}_f1 {}\n".format(metric, " ".join("{:.4f}".format(i) for i in val_avg_results[metric])))


if __name__ == '__main__':

    if args.dataset == 'reddit':
        num_labeled_list = [i for i in range(2000, 20001, 2000)]
    elif args.dataset == 'cora':
        num_labeled_list = [i for i in range(10,141,10)]
    elif args.dataset == 'citeseer':
        num_labeled_list = [i for i in range(10,121,10)]
    elif args.dataset == 'pubmed':
        num_labeled_list = [i for i in range(10,61,10)]
    elif args.dataset == 'CS':
        num_labeled_list = [i for i in range(10,151,10)]
    elif args.dataset == 'Physics':
        num_labeled_list = [i for i in range(10,101,10)]
    num_interval = len(num_labeled_list)

    val_results = {'micro': [[] for _ in range(num_interval)],
                   'macro': [[] for _ in range(num_interval)]}
    test_results = {'micro': [[] for _ in range(num_interval)],
                    'macro': [[] for _ in range(num_interval)]}
    if args.file_io:
        input_file = 'random_seed_10.txt'
        with open(input_file, 'r') as f:
            seeds = f.readline()
        seeds = list(map(int, seeds.split(' ')))
    else:
        seeds = [52, 574, 641, 934, 12]
        # seeds = [574]

    seeds = seeds * 10 # 10 runs
    seed_idx_map = {i: idx for idx, i in enumerate(seeds)}
    num_run = len(seeds)
    wrapper = run_wrapper(args.dataset, args.normalization, args.cuda)
    avg_classes_dict = None
    total_AL_time = 0
    for i in range(len(seeds)):
        print('current seed is {}'.format(seeds[i]))
        val_dict, test_dict, classes_dict, cur_AL_time = wrapper.run(args.strategy, num_labeled_list=num_labeled_list,
                                                                     seed=seeds[i])
        for metric in ['micro', 'macro']:
            for j in range(len(val_dict[metric])):
                val_results[metric][j].append(val_dict[metric][j])
                test_results[metric][j].append(test_dict[metric][j])

        total_AL_time += cur_AL_time

    val_avg_results = {'micro': [0. for _ in range(num_interval)],
                       'macro': [0. for _ in range(num_interval)]}
    test_avg_results = {'micro': [0. for _ in range(num_interval)],
                    'macro': [0. for _ in range(num_interval)]}
    val_std_results = {'micro': [0. for _ in range(num_interval)],
                        'macro': [0. for _ in range(num_interval)]}
    test_std_results = {'micro': [0. for _ in range(num_interval)],
                        'macro': [0. for _ in range(num_interval)]}
    for metric in ['micro', 'macro']:
        for j in range(len(val_results[metric])):
            val_avg_results[metric][j] = np.mean(val_results[metric][j])
            test_avg_results[metric][j] = np.mean(test_results[metric][j])
            val_std_results[metric][j] = np.std(val_results[metric][j])
            test_std_results[metric][j] = np.std(test_results[metric][j])

    if args.file_io:
        if args.model == 'GCN':
            dir_path = os.path.join('./10splits_10runs_results', args.dataset)
        else:
            dir_path = os.path.join('./results', args.dataset)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        file_path = os.path.join(dir_path, '{}.txt'.format(args.strategy))
        with open(file_path, 'a') as f:
            f.write('---------datetime: %s-----------\n' % datetime.datetime.now())
            f.write(f'Budget list: {num_labeled_list}\n')
            f.write(f'learning rate: {args.lr}, epoch: {args.epochs}, weight decay: {args.weight_decay}, hidden: {args.hidden}\n')
            f.write(f'50runs using seed.txt\n')
            for metric in ['micro', 'macro']:
                f.write("Test_{}_f1 {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_avg_results[metric])))
                f.write("Test_{}_std {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_std_results[metric])))

            f.write("Average AL_Time: {}s\n".format(total_AL_time / len(seeds)))
    else:
        print_avg_results(val_avg_results, test_avg_results)
        print("Average AL_Time: {}s\n".format(total_AL_time / len(seeds)))
