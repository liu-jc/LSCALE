import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
from collections import Counter, defaultdict
from heapq import nlargest
from torch_geometric.datasets import Amazon, Coauthor
from torch_geometric.utils.convert import to_networkx
from sklearn.metrics import pairwise_distances

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def get_one_split_random(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    idx_train, idx_val = random_resplit(idx_train, idx_val, labels)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # print('no. test: {}, train: {}, val: {}'.format(len(idx_test), len(idx_train), len(idx_val)))
    # print('idx_train: ', idx_train)
    # print('idx_val: ', idx_val)
    # num_labels = Counter()
    # for idx in idx_train:
    #     num_labels[int(labels[idx])] += 1
    # print('train set:')
    # print('label statistic: ', num_labels)

    # num_labels = Counter()
    # for idx in idx_val:
    #     num_labels[int(labels[idx])] += 1
    # print('val set:')
    # print('label statistic: ', num_labels)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def load_citation_one_step(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_non_test = list(idx_train) + list(idx_val)
    # print('idx_train: ', idx_train)
    # print('x.shape is {}'.format((x.shape)))
    # print('allx.shape is {}'.format(allx.shape))
    # print('tx.shape is {}'.format(tx.shape))
    # print('ally: ', ally)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        # idx_train = idx_train.cuda()
        # idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, graph, features, labels, idx_test, idx_non_test

def load_citation_multi_steps(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    # print('allx: ', allx)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)
    idx_non_test = list(range(len(ally)))

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        # idx_train = idx_train.cuda()
        # idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, graph, features, labels, idx_test, idx_non_test

def load_citation_laplacian(dataset_str="cora", normalization="NormLap"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    get_laplacian = fetch_normalization(normalization)
    laplacian = get_laplacian(adj)
    return laplacian

def load_citation_multi_steps_standard_split(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    # print('allx: ', allx)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)
    idx_non_test = list(range(len(ally)))

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        # idx_train = idx_train.cuda()
        # idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, graph, features, labels, idx_test, idx_non_test



def resplit_train_val(idx_train, idx_val, labels):
    labels = np.array(labels).argmax(axis=1)
    all_idx = list(idx_train) + list(idx_val)
    # print(all_idx)
    label_dict = defaultdict(list)
    # print(labels)
    for idx in all_idx:
        label_dict[int(labels[idx])].append(idx)
    # print(label_dict)
    new_idx_train = []
    new_idx_val = []
    for k, v in label_dict.items():
        sub_idx_train = np.random.choice(v, size=20, replace=False)
        new_idx_train += list(sub_idx_train)
        idx_set = set(v)
        new_idx_val += list(idx_set - set(sub_idx_train))

    num_labels = Counter()
    for idx in new_idx_train:
        num_labels[labels[idx]] += 1
    print(num_labels)
    return new_idx_train, new_idx_val


def random_resplit(idx_train, idx_val, labels):
    labels = np.array(labels).argmax(axis=1)
    all_idx = list(idx_train) + list(idx_val)
    # print(all_idx)
    # label_dict = defaultdict(list)
    # print(labels)
    # for idx in all_idx:
    #     label_dict[int(labels[idx])].append(idx)
    # print(label_dict)
    new_idx_train = []
    new_idx_val = []
    # for k, v in label_dict.items():
    #     sub_idx_train = np.random.choice(v, size=20, replace=False)
    #     new_idx_train += list(sub_idx_train)
    #     idx_set = set(v)
    #     new_idx_val += list(idx_set - set(sub_idx_train))
    #
    new_idx_train += list(np.random.choice(all_idx, size=20*7, replace=False))
    new_idx_val += list(set(all_idx) - set(new_idx_train))

    num_labels = Counter()
    for idx in new_idx_train:
        num_labels[labels[idx]] += 1
    print(num_labels)
    return new_idx_train, new_idx_val


def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time


def convert_edge2adj(edge_index, num_nodes):
    # float type
    mat = torch.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        x, y = edge_index[:, i]
        mat[x, y] = mat[y, x] = 1
    return mat

def load_Amazon(dataset_name, normalization="AugNormAdj", cuda=True):
    assert dataset_name in ['Computers', 'Photo']
    dataset = Amazon(root='./data/{}'.format(dataset_name), name='{}'.format(dataset_name))
    data = dataset[0]
    num_nodes = len(data.y)
    idx_test = [int(item) for item in open(f'./data/{dataset_name}/test_idxs.txt', 'r').readlines()]
    idx_non_test = list(set([i for i in range(num_nodes)]) - set(idx_test))
    features = data.x
    labels = data.y
    adj = convert_edge2adj(data.edge_index, num_nodes)

    adj = sp.csr_matrix(adj)
    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features)).float()
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    graph = to_networkx(data)
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

        idx_test = idx_test.cuda()

    return adj, graph, features, labels, idx_test, idx_non_test

# TODO: change the function for coauthor dataset.
def load_coauthor(dataset_name, normalization="AugNormAdj", cuda=True):
    assert dataset_name in ['CS', 'Physics']
    dataset = Coauthor(root='./data/Coauthor_{}'.format(dataset_name), name='{}'.format(dataset_name))
    data = dataset[0]
    num_nodes = len(data.y)
    idx_test = [int(item) for item in open(f'./data/Coauthor_{dataset_name}/test_idxs.txt', 'r').readlines()]
    idx_non_test = list(set([i for i in range(num_nodes)]) - set(idx_test))
    features = data.x
    labels = data.y
    adj = convert_edge2adj(data.edge_index, num_nodes)

    adj = sp.csr_matrix(adj)
    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features)).float()
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    graph = to_networkx(data)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

        idx_test = idx_test.cuda()

    return adj, graph, features, labels, idx_test, idx_non_test

def load_coauthor_SGC(dataset_name, normalization="AugNormAdj", cuda=True):
    assert dataset_name in ['CS', 'Physics']
    dataset = Coauthor(root='./data/Coauthor_{}'.format(dataset_name), name='{}'.format(dataset_name))
    data = dataset[0]
    num_nodes = len(data.y)
    idx_test = [int(item) for item in open(f'./data/Coauthor_{dataset_name}/test_idxs.txt', 'r').readlines()]
    idx_non_test = np.array(list(set([i for i in range(num_nodes)]) - set(idx_test)))
    perm = np.random.permutation(len(idx_non_test))
    num_train = 20 * (max(data.y)+1)
    print(f'num_train: {num_train}')
    idx_val = idx_non_test[perm[:500]]
    idx_train = idx_non_test[perm[500:500+num_train]]

    features = data.x
    labels = data.y
    adj = convert_edge2adj(data.edge_index, num_nodes)

    adj = sp.csr_matrix(adj)
    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features)).float()
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test

def load_Coauthor_laplacian(dataset_name, normalization="NormLap"):
    assert dataset_name in ['CS', 'Physics']
    dataset = Coauthor(root='./data/Coauthor_{}'.format(dataset_name), name='{}'.format(dataset_name))
    data = dataset[0]
    num_nodes = len(data.y)
    idx_test = [int(item) for item in open(f'./data/Coauthor_{dataset_name}/test_idxs.txt', 'r').readlines()]
    idx_non_test = np.array(list(set([i for i in range(num_nodes)]) - set(idx_test)))

    adj = convert_edge2adj(data.edge_index, num_nodes)

    adj = sp.csr_matrix(adj)
    get_laplacian = fetch_normalization(normalization)
    laplacian = get_laplacian(adj)
    return laplacian

def load_Amazon_SGC(dataset_name, normalization="AugNormAdj", cuda=True):
    assert dataset_name in ['Computers', 'Photo']
    dataset = Amazon(root='./data/{}'.format(dataset_name), name='{}'.format(dataset_name))
    data = dataset[0]
    num_nodes = len(data.y)
    idx_test = [int(item) for item in open(f'./data/{dataset_name}/test_idxs.txt', 'r').readlines()]
    idx_non_test = np.array(list(set([i for i in range(num_nodes)]) - set(idx_test)))
    perm = np.random.permutation(len(idx_non_test))
    num_train = 20 * (max(data.y)+1)
    print(f'num_train: {num_train}')
    idx_val = idx_non_test[perm[:500]]
    idx_train = idx_non_test[perm[500:500+num_train]]

    features = data.x
    labels = data.y
    adj = convert_edge2adj(data.edge_index, num_nodes)

    adj = sp.csr_matrix(adj)
    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features)).float()
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")
    print('data.keys(): {}'.format([k for k in data.keys()]))
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    print('train_idx: ', train_index)
    non_test_index = np.append(train_index, val_index)
    graph = nx.from_scipy_sparse_matrix(adj)
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, graph, train_adj, features, labels, test_index, non_test_index


def get_classes_statistic(ally):
    classes_dict = defaultdict(int)
    # ally = np.argmax(ally, axis=1)  # to index
    for y in ally:
        classes_dict[y] += 1
    classes_dict = dict(classes_dict)
    for k in classes_dict.keys():
        classes_dict[k] = classes_dict[k] / len(ally)
    # return sorted(classes_dict.items(), key= lambda x:(x[1]))
    return classes_dict


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=5):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        # print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def get_walks_stats(walks):
    node_stats = defaultdict(int)
    for walk in walks:
        for node in walk:
            node_stats[node] += 1
    return node_stats


def get_max_hitted_node(node_stats):
    return max(node_stats, key=node_stats.get)


def get_topk_hitted_node(node_stats, number):
    # counter = Counter(node_stats)
    # output = []
    # print('topk selecting... ')
    # for k, v in counter.most_common(number):
    #     print('k: {}, v: {}'.format(k, v))
    #     output.append(k)
    output = nlargest(number, node_stats, key=node_stats.get)
    for k in output:
        print('k: {}, v: {}'.format(k, node_stats[k]))
    return output


def remove_nodes_from_walks(walks, nodes):
    print('len(walks): ', len(walks))
    new_walks = []
    # print('len(new_walks): ', len(new_walks))
    for idx, walk in enumerate(walks):
        remove_flag = False
        for node in nodes:
            if node in walk:
                remove_flag = True
                break
        if not remove_flag:
            new_walks.append(walk)
    return new_walks


# calculate the percentage of elements smaller than the k-th element
def percentage_smaller(input, k):
    return sum([1 if input[i] < input[k] else 0 for i in input.keys()])/float(len(input.keys()))

# calculate the percentage of elements larger than the k-th element
def percentage_larger(input, k):
    return sum([1 if input[i] > input[k] else 0 for i in input.keys()])/float(len(input.keys()))

