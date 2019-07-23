import random
import numpy as np
from scipy import stats


def safe_div(x, y):
    if y == 0:
        return 0.0
    return round(x / y, 4)


def getProductWeight(prod_list, wallet_dist_name):
    price_list = [prod[2] for prod in prod_list]
    pw_list = [1.0 for _ in range(len(price_list))]
    if wallet_dist_name in ['m50e25', 'm99e96']:
        mu, sigma = 0, 1
        if wallet_dist_name == 'm50e25':
            mu = np.mean(price_list)
            sigma = (max(price_list) - mu) / 0.6745
        elif wallet_dist_name == 'm99e96':
            mu = sum(price_list)
            sigma = abs(min(price_list) - mu) / 3
        X = np.arange(0, 2, 0.001)
        Y = stats.norm.sf(X, mu, sigma)
        pw_list = [round(float(Y[np.argwhere(X == p)]), 4) for p in price_list]

    return pw_list


class Diffusion:
    def __init__(self, graph_dict, product_list, product_weight_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the list to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        ### monte: (int) monte carlo times
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.prob_threshold = 0.001

    def getSeedSetProfit(self, s_set):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        ep = 0.0
        for k in range(self.num_product):
            a_n_set = s_total_set.copy()
            benefit = self.product_list[k][0]
            product_weight = self.product_weight_list[k]

            i_seq = [s for s in s_set[k] if s in self.graph_dict]
            i_seq = [(i, self.graph_dict[s][i]) for s in i_seq for i in self.graph_dict[s]
                     if self.graph_dict[s][i] > max(random.random(), self.prob_threshold) and i not in a_n_set]
            while i_seq:
                # -- purchasing --
                ep += benefit * product_weight * len(i_seq)
                a_n_set = a_n_set.union(set(i[0] for i in i_seq))

                i_seq = [i for i in i_seq if i[0] in self.graph_dict]
                i_seq = [(ii, round(i[1] * self.graph_dict[i[0]][ii], 4)) for i in i_seq for ii in self.graph_dict[i[0]]
                         if round(i[1] * self.graph_dict[i[0]][ii], 4) > max(random.random(), self.prob_threshold) and ii not in a_n_set]

        return round(ep, 4)


class Evaluation:
    def __init__(self, graph_dict, prod_list):
        ### graph_dict: (dict) the graph
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_product: (int) the kinds of products
        ### wpiwp: (bool) whether passing the information with purchasing
        self.graph_dict = graph_dict
        self.product_list = prod_list
        self.num_product = len(prod_list)

    def getSeedSetProfit(self, s_set, wallet_dict):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        pro_k_list = [0.0 for _ in range(self.num_product)]
        a_n_set = [s_total_set.copy() for _ in range(self.num_product)]
        i_seq, i_seq2 = [(k, s, 1) for k in range(self.num_product) for s in s_set[k] if s in self.graph_dict], []

        while i_seq:
            k_prod, i_node, i_acc_prob = i_seq.pop(random.choice([i for i in range(len(i_seq))]))
            benefit, price = self.product_list[k_prod][0], self.product_list[k_prod][2]

            i_node_seq = [(k_prod, ii_node, round(i_acc_prob * self.graph_dict[i_node][ii_node], 4)) for ii_node in self.graph_dict[i_node]
                          if round(i_acc_prob * self.graph_dict[i_node][ii_node], 4) >= random.random() and ii_node not in a_n_set[k_prod]
                          and wallet_dict[ii_node] >= price]
            pro_k_list[k_prod] += benefit * len(i_node_seq)
            i_node_set = set(i[1] for i in i_node_seq)
            a_n_set[k_prod] = a_n_set[k_prod].union(i_node_set)
            wallet_dict = {i: wallet_dict[i] - price * (i in i_node_set) for i in wallet_dict}

            if not i_seq:
                i_seq, i_seq2 = i_seq2, i_seq

        pro_k_list = [round(pro_k, 4) for pro_k in pro_k_list]
        pnn_k_list = [len(a_n_set[k]) - len(s_total_set) for k in range(self.num_product)]

        return pro_k_list, pnn_k_list