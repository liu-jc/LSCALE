import random
from heapq import nlargest, nsmallest

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from time import perf_counter
import torch_geometric.utils as tgu
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

from utils import get_max_hitted_node, get_walks_stats, remove_nodes_from_walks, percentage_larger, percentage_smaller


def get_ppr_score(nx_G, original_weights, selected_nodes, nodes_idx):
    weights = original_weights.copy()
    for node in selected_nodes:
        weights[node] = 1./len(selected_nodes)
    PPR_scores = nx.pagerank(nx_G, alpha=0.85, personalization=weights)
    total_scores = {}
    for node in nodes_idx:
        # total_scores[node] = PPR_scores[node] - PR_scores[node]
        total_scores[node] = PPR_scores[node]
    return total_scores

def get_uncertainty_score(model, features, nodes_idx):
    model.eval()
    output = model(features[nodes_idx])
    prob_output = F.softmax(output, dim=1).detach()
    # log_prob_output = torch.log(prob_output).detach()
    log_prob_output = F.log_softmax(output, dim=1).detach()
    # print('prob_output[0]: ', prob_output[0])
    # print('log_prob_output[0]: ', log_prob_output[0])
    entropy = -torch.sum(prob_output*log_prob_output, dim=1)  # contain node idx
    entropy = entropy.cpu().numpy()
    total_score = {}
    for idx, node in enumerate(nodes_idx):
        total_score[node] = entropy[idx]
    return total_score

def query_unified(nx_G, model, features, selected_nodes, number, pool):
    # print('number is {}'.format(number))
    nodes = nx.nodes(nx_G)
    uncertainty_score = get_uncertainty_score(model, features, nodes)
    weights = {n: float(uncertainty_score[n]) for n in nodes}
    for node in selected_nodes:
        weights[node] = 0.
    # weights = {n: 1./len(nodes) for n in nodes}
    # print('weights: ', weights)
    # print('the uncertainty scores of selected nodes:', [uncertainty_score[node] for node in selected_nodes])
    # print('uncertainty scores (increasing order): ', sorted(uncertainty_score.values())[:100])
    # print('selected nodes: ', selected_nodes)
    # print('scores: ', [uncertainty_score[node] for node in selected_nodes])
    # n_smallest_list = nsmallest(len(selected_nodes), uncertainty_score, key=uncertainty_score.get)
    # print('n_smallest_list: ', n_smallest_list)
    # print('scores: ', [uncertainty_score[node] for node in n_smallest_list])
    # hit_rate = np.sum([1 if node in n_smallest_list else 0 for node in selected_nodes]) / len(selected_nodes)
    # print('hit_rate: ', hit_rate)
    new_weights = weights.copy()
    # for k in selected_nodes:
    #     new_weights[k] = new_weights[k] - 1./len(selected_nodes) # change the teleport set.
    # for k in nodes:
    #     new_weights[k] += 1./len(selected_nodes)
    unified_scores = nx.pagerank(nx_G, alpha=0.85, personalization=new_weights)
    total_scores = {}
    for node in pool:
        total_scores[node] = unified_scores[node]
    # topk_scores = {k: v for k, v in sorted(total_scores.items(), key=lambda item: item[1])}
    idx_topk = nlargest(number, total_scores, key=total_scores.get)
    # print('topk_scores: ', [v for k,v in list(topk_scores.items())[:number]])
    # return list(total_scores.keys())[idx_topk]
    return idx_topk, weights

def query_uncertainty_pr_ppr(model, feature, nx_G, original_weights, PR_scores, selected_nodes, number, nodes_idx):
    # nodes = nx.nodes(nx_G)
    # entropy_score = get_uncertainty_score(model, feature, nodes)
    entropy_score = get_uncertainty_score(model, feature, nodes_idx)
    PPR_scores = get_ppr_score(nx_G, original_weights, selected_nodes, nodes_idx)
    total_scores = {}
    max_PR = -1
    for k in PR_scores.keys():
        max_PR = max(max_PR, PR_scores[k])
    max_PPR = -1
    for k in PPR_scores.keys():
        max_PPR = max(max_PPR, PPR_scores[k])
    max_entropy = -10000
    for k in entropy_score.keys():
        max_entropy = max(max_entropy, entropy_score[k])
    # for node in nodes_idx:
    #     total_scores[node] = PR_scores[node] - PPR_scores[node] + entropy_score[node]
    # uncertainty_score = entropy_score
    # print('selected nodes: ', selected_nodes)
    # print('scores: ', [uncertainty_score[node] for node in selected_nodes])
    # n_smallest_list = nsmallest(len(selected_nodes), uncertainty_score, key=uncertainty_score.get)
    # print('n_smallest_list: ', n_smallest_list)
    # print('scores: ', [uncertainty_score[node] for node in n_smallest_list])
    # hit_rate = np.sum([1 if node in n_smallest_list else 0 for node in selected_nodes]) / len(selected_nodes)
    # print('hit_rate: ', hit_rate)
    for node in nodes_idx:
        total_scores[node] = PR_scores[node] / max_PR - PPR_scores[node] / max_PPR + entropy_score[node] / max_entropy
    # idx_topk = nlargest(2*number, total_scores, key=total_scores.get)
    # idx_topk = np.random.choice(idx_topk, size=number, replace=False)
    idx_topk = nlargest(number, total_scores, key=total_scores.get)
    # print('pr_scores: ', [PR_scores[k] for k in idx_topk])
    # print('ppr_scores: ', [PPR_scores[k] for k in idx_topk])
    # print('entropy_score: ', [entropy_score[k] for k in idx_topk])
    # print('topk_scores: ', [total_scores[k] for k in idx_topk])
    return idx_topk

    # entropy_score = get_uncertainty_score(model, feature, nodes_idx)
    # idx_topk = []
    # for i in range(number):
    #     weights = original_weights.copy()
    #     for node in selected_nodes:
    #         weights[node] = 1./len(selected_nodes)
    #     PPR_scores = nx.pagerank(nx_G, alpha=0.85, personalization=weights)
    #     total_scores = {}
    #     max_id = -1
    #     max_value = -10000
    #     for node in nodes_idx:
    #         total_scores[node] = PR_scores[node] - PPR_scores[node] + entropy_score[node]
    #         if total_scores[node] > max_value:
    #             max_value = total_scores[node]
    #             max_id = node
    #     idx_topk.append(max_id)

def query_mixed_random(model, feature, nx_G, original_weights, PR_scores, selected_nodes, number, nodes_idx):
    entropy_score = get_uncertainty_score(model, feature, nodes_idx)
    PPR_scores = get_ppr_score(nx_G, original_weights, selected_nodes, nodes_idx)
    total_scores = {}
    max_PR = -1
    for k in PR_scores.keys():
        max_PR = max(max_PR, PR_scores[k])
    max_PPR = -1
    for k in PPR_scores.keys():
        max_PPR = max(max_PPR, PPR_scores[k])
    max_entropy = -10000
    for k in entropy_score.keys():
        max_entropy = max(max_entropy, entropy_score[k])

    for node in nodes_idx:
        total_scores[node] = PR_scores[node] / max_PR - PPR_scores[node] / max_PPR + entropy_score[node] / max_entropy
    idx_topk = nlargest(2*number, total_scores, key=total_scores.get)
    idx_topk = np.random.choice(idx_topk, size=number, replace=False)
    # idx_topk = nlargest(number, total_scores, key=total_scores.get)
    print('pr_scores: ', [PR_scores[k] for k in idx_topk])
    print('ppr_scores: ', [PPR_scores[k] for k in idx_topk])
    print('entropy_score: ', [entropy_score[k] for k in idx_topk])
    print('topk_scores: ', [total_scores[k] for k in idx_topk])
    return idx_topk


def query_random(number, nodes_idx):
    return np.random.choice(nodes_idx, size=number, replace=False)


def query_largest_degree(nx_graph, number, nodes_idx):
    degree_dict = nx_graph.degree(nodes_idx)
    idx_topk = nlargest(number, degree_dict, key=degree_dict.get)
    # print(idx_topk)
    return idx_topk


def query_uncertainty(model, features, number, nodes_idx):
    model.eval()
    output = model(features[nodes_idx])
    prob_output = F.softmax(output, dim=1).detach()
    # log_prob_output = torch.log(prob_output).detach()
    log_prob_output = F.log_softmax(output, dim=1).detach()
    # print('prob_output: ', prob_output)
    # print('log_prob_output: ', log_prob_output)
    entropy = -torch.sum(prob_output*log_prob_output, dim=1)
    # print('entropy: ', entropy)
    indices = torch.topk(entropy, number, largest=True)[1]
    # print('indices: ', list(indices.cpu().numpy()))
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]
    # return indices

def query_uncertainty_GCN(model, adj, features, number, nodes_idx):
    model.eval()
    # output = model(features[nodes_idx])
    output = model(features, adj)
    output = output[nodes_idx, :]
    prob_output = F.softmax(output, dim=1).detach()
    # log_prob_output = torch.log(prob_output).detach()
    log_prob_output = F.log_softmax(output, dim=1).detach()
    # print('prob_output: ', prob_output)
    # print('log_prob_output: ', log_prob_output)
    entropy = -torch.sum(prob_output*log_prob_output, dim=1)
    # print('entropy: ', entropy)
    indices = torch.topk(entropy, number, largest=True)[1]
    # print('indices: ', list(indices.cpu().numpy()))
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]

def query_random_uncertainty(model, features, number, nodes_idx):
    model.eval()
    output = model(features[nodes_idx])
    prob_output = F.softmax(output, dim=1).detach()
    log_prob_output = torch.log(prob_output).detach()
    entropy = -torch.sum(prob_output*log_prob_output, dim=1)
    indices = torch.topk(entropy, 3*number, largest=True)[1]
    indices = np.random.choice(indices.cpu().numpy(), size=number, replace=False)
    return np.array(nodes_idx)[indices]


def qeury_coreset_greedy(features, selected_nodes, number, nodes_idx):
    features = features.cpu().numpy()
    # print('nodes_idx: ', nodes_idx)
    def get_min_dis(features, selected_nodes, nodes_idx):
        Y = features[selected_nodes]
        X = features[nodes_idx]
        dis = pairwise_distances(X, Y)
        # print('dis: ', dis)
        return np.min(dis, axis=1)

    new_batch = []
    for i in range(number):
        if selected_nodes == []:
            ind = np.random.choice(nodes_idx)
        else:
            min_dis = get_min_dis(features, selected_nodes, nodes_idx)
            # print('min_dis: ', min_dis)
            ind = np.argmax(min_dis)
        # print('ind: ', ind)
        assert nodes_idx[ind] not in selected_nodes

        selected_nodes.append(nodes_idx[ind])
        new_batch.append(nodes_idx[ind])
        # print('%d item: %d' %(i, nodes_idx[ind]))
    return np.array(new_batch)


def query_AGE():
    pass


def query_featprop(features, number, nodes_idx):
    features = features.cpu().numpy()
    X = features[nodes_idx]
    # print('X: ', X)
    t1 = perf_counter()
    distances = pairwise_distances(X, X)
    print('computer pairwise_distances: {}s'.format(perf_counter() - t1))
    clusters, medoids = k_medoids(distances, k=number)
    # print('cluster: ', clusters)
    # print('medoids: ', medoids)
    # print('new indices: ', np.array(nodes_idx)[medoids])
    return np.array(nodes_idx)[medoids]

def query_new_featprop(features, num_points, nodes_idx):
    from pyclustering.cluster.kmedoids import kmedoids
    from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

    features = features.cpu().numpy()
    start_time = perf_counter()
    # Prepare initial centers using K-Means++ method.
    initial_centers = kmeans_plusplus_initializer(features, num_points).initialize() # num_points x feature_dim
    distances = pairwise_distances(features, initial_centers, n_jobs=-1) # parallel computing, n x num_points
    initial_medoids = np.argmin(distances, axis=0)
    print('Medoids number', len(initial_medoids))
    # Create instance of K-Medoids algorithm.
    kmedoids_instance = kmedoids(features, initial_medoids)
    # Run cluster analysis and obtain results.
    kmedoids_instance.process()
    print('K-Medoids clustering time', perf_counter() - start_time)
    full_new_index_list = kmedoids_instance.get_medoids()
    return np.array(full_new_index_list)


def query_featprop_increment(features, number, fixed_medoids, nodes_idx):
    features = features.cpu().numpy()
    X = features[nodes_idx]
    # X = features
    # print('X: ', X)
    t1 = perf_counter()
    distances = pairwise_distances(X, X)
    print('computer pairwise_distances: {}s'.format(perf_counter() - t1))
    clusters, medoids = new_k_medoids(fixed_medoids, distances, k=number)
    # print('cluster: ', clusters)
    # print('medoids: ', medoids)
    # print('new indices: ', np.array(nodes_idx)[medoids])
    return np.array(nodes_idx)[medoids], medoids
    # return np.array(medoids)

def dis_ppr(X, adj, alpha, k):
    X_0 = X
    for i in range(k):
        X = (1-alpha) * torch.sparse.mm(adj, X) + alpha * X_0
    return X

def reweight(features, model):
    W = model.W.weight
    W_l2 = torch.norm(W, dim=0).detach()
    # W_l2 = F.softmax(W_l2)
    # print(f'W_l2: {W_l2}')
    # print(f'features.size(): {features.size()}')
    # input('reweights')
    return features * W_l2

def query_ours(dis_features, model, number, nodes_idx, reweight_flag=True):
    if reweight_flag:
        dis_features = reweight(dis_features, model)
    return query_featprop(dis_features, number, nodes_idx)


def query_ours_increment(dis_features, model, number, fixed_medoids, nodes_idx, reweight_flag=True):
    if reweight_flag:
        dis_features = reweight(dis_features, model)
    return query_featprop_increment(dis_features, number, fixed_medoids, nodes_idx)


def query_ppr(nx_G, original_weights, selected_nodes, PR_scores, number, nodes_idx):
    weights = original_weights.copy()
    for node in selected_nodes:
        weights[node] = 1./len(selected_nodes)
    PPR_scores = nx.pagerank(nx_G, alpha=0.85, personalization=weights)
    total_scores = {}
    for node in nodes_idx:
        # total_scores[node] = PPR_scores[node] - PR_scores[node]
        total_scores[node] = PPR_scores[node]
    topk_scores = {k: v for k, v in sorted(total_scores.items(), key=lambda item: item[1])}
    # print('ppr_scores: ', PPR_scores)
    # print('topk_scores: ', topk_scores)
    # for key in topk_scores.keys():
    #     print('ppr[{}]: {}, PR_scores[{}]: {}'.format(key, PPR_scores[key], key, PR_scores[key]))
    print('topk_scores: ', [v for k,v in list(topk_scores.items())[:number]])
    return list(topk_scores.keys())[:number]


def query_pr_ppr(nx_G, original_weights, selected_nodes, PR_scores, number, nodes_idx):
    weights = original_weights.copy()
    for node in selected_nodes:
        weights[node] = 1./len(selected_nodes)
    PPR_scores = nx.pagerank(nx_G, alpha=0.85, personalization=weights)
    total_scores = {}
    for node in nodes_idx:
        total_scores[node] = PR_scores[node] - PPR_scores[node]
    topk_scores = {k: v for k, v in sorted(total_scores.items(), key=lambda item: item[1])}
    # print('ppr_scores: ', PPR_scores)
    # print('topk_scores: ', topk_scores)
    # for key in topk_scores.keys():
    #     print('ppr[{}]: {}, PR_scores[{}]: {}'.format(key, PPR_scores[key], key, PR_scores[key]))
    print('topk_scores: ', [v for k, v in list(topk_scores.items())[-number:]])
    return list(topk_scores.keys())[-number:]


def query_pr(PR_scores, number, nodes_idx):
    selected_scores = {}
    for node in nodes_idx:
        selected_scores[node] = PR_scores[node]
    topk_scores = {k: v for k, v in sorted(selected_scores.items(), key = lambda item: item[1])}
    # print('ppr_scores: ', PPR_scores)
    # print('topk_scores: ', topk_scores)
    # for key in topk_scores.keys():
    #     print('ppr[{}]: {}, PR_scores[{}]: {}'.format(key, PPR_scores[key], key, PR_scores[key]))
    # print('tok_scores: ', [v for k,v in list(topk_scores.items())[-number:]])
    return list(topk_scores.keys())[-number:]


def query_rw(walks, number, nodes_idx):
    topk_nodes = []
    for i in range(number):
        node_stats = get_walks_stats(walks)
        top1_node = get_max_hitted_node(node_stats)
        # print('top1 node {}: {}'.format(top1_node, node_stats[top1_node]))
        topk_nodes.append(top1_node)
        walks = remove_nodes_from_walks(walks, [top1_node])
    return walks, topk_nodes
    # nodes_stats = get_walks_stats(walks)
    # topk_nodes = get_topk_hitted_node(nodes_stats, number)
    # return topk_nodes

def new_k_medoids(fixed_medoids, distances, k=3):
    # From https://github.com/salspaugh/machine_learning/blob/master/clustering/kmedoids.py

    m = distances.shape[0] # number of points

    # Pick k random medoids.
    print('k: {}'.format(k))
    # curr_medoids = np.array([-1]*k)
    # while not len(np.unique(curr_medoids)) == k:
    #     curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    fixed_num = len(fixed_medoids)
    # curr_medoids = np.arange(m)
    # np.random.shuffle(curr_medoids)
    # curr_medoids = curr_medoids[:k]
    candidates = np.array(list(set(np.arange(m)) - set(fixed_medoids)))
    curr_medoids = np.random.choice(candidates, size=k, replace=False)
    curr_medoids[:fixed_num] = fixed_medoids
    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)
    new_medoids[:fixed_num] = fixed_medoids

    # Until the medoids stop updating, do the following:
    num_iter = 0
    while not ((old_medoids == curr_medoids).all()):
        num_iter += 1
        # print('curr_medoids: ', curr_medoids)
        # Assign each point to cluster with closest medoid.
        t1 = perf_counter()
        clusters = assign_points_to_clusters(curr_medoids, distances)
        # print(f'clusters: {clusters}')
        # print('time assign point ot clusters: {}s'.format(perf_counter() - t1))
        # Update cluster medoids to be lowest cost point.
        t1 = perf_counter()
        for idx, curr_medoid in enumerate(curr_medoids):
            # print(f'idx: {idx}')
            if idx < fixed_num:
                continue
            cluster = np.where(clusters == curr_medoid)[0]
            # cluster = np.asarray(clusters == curr_medoid)
            # print(f'curr_medoid: {curr_medoid}')
            # print(f'np.where(clusters == curr_medoid): {np.where(clusters == curr_medoid)}')
            # print(f'cluster: {cluster}')
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)
            del cluster
        # print('time update medoids: {}s'.format(perf_counter() - t1))
        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
    print('total num_iter is {}'.format(num_iter))
    # print(f'curr_medoids: {curr_medoids}')
    # input('wait')
    print('-----------------------------')
    return clusters, curr_medoids[fixed_num:]

def k_medoids(distances, k=3):
    # From https://github.com/salspaugh/machine_learning/blob/master/clustering/kmedoids.py

    m = distances.shape[0] # number of points

    # Pick k random medoids.
    print('k: {}'.format(k))
    # curr_medoids = np.array([-1]*k)
    # while not len(np.unique(curr_medoids)) == k:
    #     curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    curr_medoids = np.arange(m)
    np.random.shuffle(curr_medoids)
    curr_medoids = curr_medoids[:k]
    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)

    # Until the medoids stop updating, do the following:
    num_iter = 0
    while not ((old_medoids == curr_medoids).all()):
        num_iter += 1
        # print('curr_medoids: ', curr_medoids)
        # print('old_medoids: ', old_medoids)
        # Assign each point to cluster with closest medoid.
        t1 = perf_counter()
        clusters = assign_points_to_clusters(curr_medoids, distances)
        # print(f'clusters: {clusters}')
        # print('time assign point ot clusters: {}s'.format(perf_counter() - t1))
        # Update cluster medoids to be lowest cost point.
        t1 = perf_counter()
        for idx, curr_medoid in enumerate(curr_medoids):
            # print(f'idx: {idx}')
            cluster = np.where(clusters == curr_medoid)[0]
            # cluster = np.asarray(clusters == curr_medoid)
            # print(f'curr_medoid: {curr_medoid}')
            # print(f'np.where(clusters == curr_medoid): {np.where(clusters == curr_medoid)}')
            # print(f'cluster: {cluster}')
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)
            del cluster
        # print('time update medoids: {}s'.format(perf_counter() - t1))
        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        if num_iter >= 50:
            print(f'Stop as reach {num_iter} iterations')
            break
    print('total num_iter is {}'.format(num_iter))
    print('-----------------------------')
    return clusters, curr_medoids


def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters


def compute_new_medoid(cluster, distances):
    # mask = np.ones(distances.shape)
    # print(f'distance[10,10]: {distances[10,10]}')
    # t1 = perf_counter()
    # mask[np.ix_(cluster,cluster)] = 0.
    # print(f'np.ix_(cluster,cluster): {np.ix_(cluster,cluster)}')
    # print(f'mask: {mask}')
    # print('time creating mask: {}s'.format(perf_counter()-t1))
    # input('before')
    # cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    # print(f'cluster_distances: {cluster_distances}')
    # t1 = perf_counter()
    # print('cluster_distances.shape: {}'.format(cluster_distances.shape))
    # costs = cluster_distances.sum(axis=1)
    # print(f'costs: {costs}')
    # print('time counting costs: {}s'.format(perf_counter()-t1))
    # print(f'medoid: {costs.argmin(axis=0, fill_value=10e9)}')
    # return costs.argmin(axis=0, fill_value=10e9)
    cluster_distances = distances[cluster,:][:,cluster]
    costs = cluster_distances.sum(axis=1)
    min_idx = costs.argmin(axis=0)
    # print(f'new_costs: {costs}')
    # print(f'new_medoid: {cluster[min_idx]}')
    return cluster[min_idx]


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)



# Base class
class ActiveLearner:
    def __init__(self, G, data):
        self.data = data
        self.n = data.num_nodes
        self.G = G
        # if prev_index is None:
        #     self.prev_index_list = []
        # else:
        #     self.prev_index_list = np.where(self.prev_index.cpu().numpy())[0]


    def choose(self, num_points):
        raise NotImplementedError

    def pretrain_choose(self, num_points):
        raise NotImplementedError

class AnrmabLearner(ActiveLearner):
    def __init__(self, G, data):
        # start_time = time.time()
        super(AnrmabLearner, self).__init__(G, data)
        self.device = data.x.get_device()

        self.y = data.y.detach().cpu().numpy()
        self.NCL = len(np.unique(data.y.cpu().numpy()))

        self.G = tgu.to_networkx(data.edge_index)
        self.normcen = centralissimo(self.G).flatten()
        self.w = np.array([1., 1., 1.]) # ie, nc, id
        # print('AnrmabLearner init time', time.time() - start_time)

    def pretrain_choose(self, model, features, adj, nodes_idx, num_points):
        # here we adopt a slightly different strategy which does not exclude sampled points in previous rounds to keep consistency with other methods
        # num_points -> budget
        # self.model.eval()
        # (features, prev_out, no_softmax), out = self.model(self.data)
        model.eval()
        # Here model should be GCN
        output = model(features, adj)
        prob_output = F.softmax(output, dim=1).detach()
        log_prob_output = F.log_softmax(output, dim=1).detach()
        scores = -torch.sum(prob_output*log_prob_output, dim=1)

        # if self.args.uncertain_score == 'entropy':
        #     scores = torch.sum(-F.softmax(prev_out, dim=1) * F.log_softmax(prev_out, dim=1), dim=1)
        # elif self.args.uncertain_score == 'margin':
        #     pred = F.softmax(prev_out, dim=1)
        #     top_pred, _ = torch.topk(pred, k=2, dim=1)
        #     # use negative values, since the largest values will be chosen as labeled data
        #     scores =  (-top_pred[:,0] + top_pred[:,1]).view(-1)
        # else:
        #     raise NotImplementedError

        # epoch = len(self.prev_index_list)

        softmax_out = F.softmax(output, dim=1).cpu().detach().numpy()
        kmeans = KMeans(n_clusters=self.NCL, random_state=0).fit(softmax_out)
        ed = euclidean_distances(softmax_out,kmeans.cluster_centers_)
        ed_score = np.min(ed,axis=1)	#the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is

        q_ie = scores.detach().cpu().numpy()
        q_nc = self.normcen
        q_id = 1. / (1. + ed_score)
        q_mat = np.vstack([q_ie, q_nc, q_id])  # 3 x n
        q_sum = q_mat.sum(axis=1, keepdims=True)
        q_mat = q_mat / q_sum

        w_len = self.w.shape[0]
        p_min = np.sqrt(np.log(w_len) / w_len / num_points)
        p_mat = (1 - w_len*p_min) * self.w / self.w.sum() + p_min # 3

        phi = p_mat[:, np.newaxis] * q_mat # 3 x n
        phi = phi.sum(axis=0) # n

        # sample new points according to phi
        # TODO: change to the sampling method
        # if self.args.anrmab_argmax:
        #     full_new_index_list = np.argsort(phi)[::-1][:num_points] # argmax
        # else:
        #     full_new_index_list = np.random.choice(len(phi), num_points, p=phi)

        full_new_index_list = np.random.choice(len(phi), num_points, p=phi)

        # mask = combine_new_old(full_new_index_list, self.prev_index_list, num_points, self.n, in_order=True)
        # mask_list = np.where(mask)[0]
        # diff_list = np.asarray(list(set(mask_list).difference(set(self.prev_index_list))))

        pred = torch.argmax(out, dim=1).detach().cpu().numpy()
        reward = 1. / num_points / (self.n - num_points) * np.sum((pred[mask_list] == self.y[mask_list]).astype(np.float) / phi[mask_list]) # scalar
        reward_hat = reward * np.sum(q_mat[:, diff_list] / phi[np.newaxis, diff_list], axis=1)
        # update self.w
        # get current node label epoch
        epoch = self.args.label_list.index(num_points) + 1
        p_const = np.sqrt(np.log(self.n * 10. / 3. / epoch))
        self.w = self.w * np.exp(p_min / 2 * (reward_hat + 1. / p_mat * p_const))

        # import ipdb; ipdb.set_trace()
        # print('Age pretrain_choose time', time.time() - start_time)

        return mask


def centralissimo(G):
    centralities = []
    centralities.append(nx.pagerank(G))                #print 'page rank: check.'
    L = len(centralities[0])
    Nc = len(centralities)
    cenarray = np.zeros((Nc,L))
    for i in range(Nc):
        cenarray[i][list(centralities[i].keys())]=list(centralities[i].values())
    normcen = (cenarray.astype(float)-np.min(cenarray,axis=1)[:,None])/(np.max(cenarray,axis=1)-np.min(cenarray,axis=1))[:,None]
    return normcen