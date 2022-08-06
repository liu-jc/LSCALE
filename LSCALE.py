import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation_multi_steps, early_stopping, remove_nodes_from_walks, sgc_precompute, \
    get_classes_statistic, load_reddit_data, load_Amazon, load_coauthor
from models import get_model, DGI
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
                    }, 'checkpoint_inc12.pt')
                if should_stop:
                    print('epoch: {}, acc_val: {}, best_acc_val: {}'.format(epoch, acc_val, best_acc_val))
                    # load best model
                    checkpoint = torch.load('checkpoint_inc12.pt')
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
        if dataset in ['Computers', 'Photo']:
            self.adj, self.graph, self.features, self.labels, self.idx_test, self.idx_non_test = load_Amazon(dataset, normalization,
                                                                                                 cuda=cuda)
        elif dataset in ['CS', 'Physics']:
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
        print('finished loading dataset')
        self.raw_features = self.features
        self.nb_nodes = self.features.size(0)
        if args.model == "SGC":
            self.features, precompute_time = sgc_precompute(self.features, self.adj, args.degree)
            print("{:.4f}s".format(precompute_time))



    def run(self, strategy, num_labeled_list=[10, 15, 20, 25, 30, 35, 40, 50], max_budget=160, seed=1):
        set_seed(seed, args.cuda)
        max_budget = num_labeled_list[-1]

        idx_non_test = self.idx_non_test.copy()
        print('len(idx_non_test) is {}'.format(len(idx_non_test)))
        # Select validation nodes.
        num_val = 500
        idx_val = np.random.choice(idx_non_test, num_val, replace=False)
        idx_non_test = list(set(idx_non_test) - set(idx_val))

        # unsupervised loss via Deep graph infomax.
        # X' = \sigma(WX') then update the weights via unsupervised loss
        batch_size = 1
        dgi_lr = 0.001
        dgi_weight_decay = 0.0
        dgi_epoch = 1000
        best_loss = 1e9
        best_iter = 0
        cnt_wait = 0
        patience = 20
        b_xent = torch.nn.BCEWithLogitsLoss()
        ft_size = self.raw_features.size(1)
        nb_nodes = self.raw_features.size(0)
        features = self.raw_features[np.newaxis]
        print("----------all Parameters-----------")
        if args.dataset in ['pubmed', 'Computers', 'Photo']:
            hidden = 256
        elif args.dataset in ['Physics', 'CS']:
            hidden = 128
        else:
            hidden = 512
        DGI_model = DGI(ft_size, hidden, 'prelu')
        for name, param in DGI_model.named_parameters():
            if param.requires_grad:
                print(name, param.size())
        opt = optim.Adam(DGI_model.parameters(), lr=dgi_lr,
                               weight_decay=dgi_weight_decay)
        DGI_model.train()
        print('Training unsupervised model.....')
        for i in range(dgi_epoch):
            opt.zero_grad()

            perm_idx = np.random.permutation(self.nb_nodes)
            shuf_fts = features[:, perm_idx, :]


            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)
            if torch.cuda.is_available():
                DGI_model.cuda()
                shuf_fts = shuf_fts.cuda()
                lbl = lbl.cuda()

            logits = DGI_model(features, shuf_fts, self.adj, True, None, None, None)

            loss = b_xent(logits, lbl)


            if loss.item() < best_loss:
                best_loss = loss.item()
                best_iter = i
                cnt_wait = 0
                torch.save(DGI_model.state_dict(), 'best_dgi_inc11.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early Stopping')
                break

            loss.backward()
            opt.step()

        print(f'Finished training unsupervised model, Loading {best_iter}th epoch')
        DGI_model.load_state_dict(torch.load('best_dgi_inc11.pkl'))
        self.features, _ = DGI_model.embed(features, self.adj, True, None)
        self.features = torch.squeeze(self.features, 0)

        # initially select some nodes.
        L = 5
        selected_nodes = np.random.choice(idx_non_test, L, replace=False)
        idx_non_test = list(set(idx_non_test) - set(selected_nodes))

        model = get_model('distance_based', self.features.size(1), self.labels.max().item() + 1, args.hidden, args.dropout,
                          args.cuda)

        # Multi-step select nodes to label
        budget = 20
        steps = 6
        pool = idx_non_test
        print('len(idx_non_test): {}'.format(len(idx_non_test)))
        np.random.seed() # cancel the fixed seed

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

        time_AL = 0
        fixed_medoids = []
        for i in range(len(num_labeled_list)):
            # if num_labeled_list[i] > max_budget:
            #     break
            # budget = num_labeled_list[i] - cur_num
            budget = num_labeled_list[i]
            u_features = model.new_features(self.features)
            if args.feature == 'cat':
                if args.adaptive == 1:
                    alpha = 0.99 ** num_labeled_list[i]
                    beta = 1 - alpha
                    print(f'alpha: {alpha}, beta: {beta}')
                    dis_features = torch.cat((alpha * F.normalize(self.features, p=1, dim=1), beta * F.normalize(u_features, p=1, dim=1)), dim=1)
                else:
                    dis_features = torch.cat((F.normalize(self.features, dim=1), F.normalize(u_features, dim=1)), dim=1)
            else:
                dis_features = u_features
            t1 = perf_counter()
            if strategy == 'LSCALE':
                idx_train, original_medoids = query_ours_increment(dis_features, model, budget, fixed_medoids, pool, reweight_flag=args.reweight)
            else:
                raise NotImplementedError('cannot find the strategy {}'.format(strategy))

            time_AL += perf_counter() - t1
            #print(f'selected_nodes: {selected_nodes}')
            #print(f'idx_train: {idx_train}')
            ensure_nonrepeat(idx_train, selected_nodes)
            selected_nodes = np.append(selected_nodes, idx_train)
            fixed_medoids.extend(original_medoids)
            #print(f'fixed_medoids: {fixed_medoids}')
            assert len(fixed_medoids) == budget

            model, acc_val, micro_val, macro_val, train_time = train_regression(model, self.features[selected_nodes],
                                                                                self.labels[selected_nodes],
                                                                                self.features[idx_val],
                                                                                self.labels[idx_val],
                                                                                args.epochs, args.weight_decay, args.lr,
                                                                                args.dropout)

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

    num_per_splits = 10
    seeds = seeds * num_per_splits
    num_run = len(seeds)
    wrapper = run_wrapper(args.dataset, args.normalization, args.cuda)
    avg_classes_dict = None
    total_AL_time = 0
    for i in range(len(seeds)):
        # val_dict, test_dict = run(args.strategy, dataset=args.dataset, seed=seeds[i])
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
        dir_path = os.path.join('./increment_clustering_10_10_results', args.dataset)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        file_path = os.path.join(dir_path, '{}.txt'.format(args.strategy))
        with open(file_path, 'a') as f:
            f.write('---------datetime: %s-----------\n' % datetime.datetime.now())
            f.write(f'Budget list: {num_labeled_list}\n')
            f.write(f'learning rate: {args.lr}, epoch: {args.epochs}, reweighting: {args.reweight}\n')
            f.write(f'incremental clustering \nNew idea, hidden: {args.hidden}, 50runs, args.feature: {args.feature}\n')
            for metric in ['micro', 'macro']:
                f.write("Test_{}_f1 {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_avg_results[metric])))
                f.write("Test_{}_std {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_std_results[metric])))

            f.write("Average AL_Time: {}s\n".format(total_AL_time / len(seeds)))
    else:
        print_avg_results(val_avg_results, test_avg_results)
        # print(avg_classes_dict)
        print("Average AL_Time: {}s\n".format(total_AL_time / len(seeds)))
