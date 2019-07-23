from Initialization import *
from Diffusion import *
import time
import os


def getTotalNumNode(graph_dict):
    node_set = set(i for i in graph_dict)
    for i in graph_dict:
        node_set = node_set.union(set(ii for ii in graph_dict[i]))

    return len(node_set)


class EvaluationM:
    def __init__(self, model_name, dataset_name, product_name, cascade_model):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.eva_monte_carlo = 100

    def evaluate(self, bi, wallet_distribution_type, seed_set, ss_time):
        eva_start_time = time.time()
        ini = Initialization(self.dataset_name, self.product_name)
        iniW = IniWallet(self.dataset_name, self.product_name, wallet_distribution_type)

        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        wallet_dict = iniW.constructWalletDict()
        num_node = getTotalNumNode(graph_dict) * num_product
        total_budget = max(safe_div(num_node, 2 ** bi), 1)

        eva = Evaluation(graph_dict, product_list)
        print('@ ' + self.model_name + ' evaluation @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model +
              ', product_name = ' + self.product_name + ', wd = ' + wallet_distribution_type)
        sample_pro_k_acc, sample_pnn_k_acc = [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]

        for _ in range(self.eva_monte_carlo):
            pro_k_list, pnn_k_list = eva.getSeedSetProfit(seed_set, wallet_dict.copy())
            sample_pro_k_acc = [(pro_k + sample_pro_k) for pro_k, sample_pro_k in zip(pro_k_list, sample_pro_k_acc)]
            sample_pnn_k_acc = [(pnn_k + sample_pnn_k) for pnn_k, sample_pnn_k in zip(pnn_k_list, sample_pnn_k_acc)]
        sample_pro_k_acc = [round(sample_pro_k / self.eva_monte_carlo, 4) for sample_pro_k in sample_pro_k_acc]
        sample_pnn_k_acc = [round(sample_pnn_k / self.eva_monte_carlo, 4) for sample_pnn_k in sample_pnn_k_acc]
        sample_bud_k_acc = [len(seed_set[k]) for k in range(num_product)]
        sample_sn_k_acc = [len(sample_sn_k) for sample_sn_k in seed_set]
        sample_pro_acc = round(sum(sample_pro_k_acc), 4)
        sample_bud_acc = round(sum(sample_bud_k_acc), 4)

        result = [sample_pro_acc, sample_bud_acc, sample_sn_k_acc, sample_pnn_k_acc, sample_pro_k_acc, sample_bud_k_acc]

        print('eva_time = ' + str(round(time.time() - eva_start_time, 2)) + 'sec')
        print(result)
        print('------------------------------------------')

        path = 'result/' + self.model_name + '_' + wallet_distribution_type
        if not os.path.isdir(path):
            os.mkdir(path)
        fw = open(path + '/' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(bi) + '.txt', 'w')
        fw.write(self.model_name + ', ' + wallet_distribution_type + ', ' + self.dataset_name + '_' + self.cascade_model + ', ' + self.product_name + '\n\n' +
                 'total_budget = ' + str(total_budget) + ', sample_number = ' + str(len(seed_set)) + '\n' +
                 'profit = ' + str(sample_pro_acc) + ', cost = ' + str(sample_bud_acc) + ', time = ' + str(ss_time) + '\n')
        fw.write('\nprofit_ratio =')
        for kk in range(num_product):
            fw.write(' ' + str(sample_pro_k_acc[kk]))
        fw.write('\nbudget_ratio =')
        for kk in range(num_product):
            fw.write(' ' + str(sample_bud_k_acc[kk]))
        fw.write('\nseed_number =')
        for kk in range(num_product):
            fw.write(' ' + str(sample_sn_k_acc[kk]))
        fw.write('\ncustomer_number =')
        for kk in range(num_product):
            fw.write(' ' + str(sample_pnn_k_acc[kk]))
        fw.write('\n\n' + str(seed_set))
        fw.close()