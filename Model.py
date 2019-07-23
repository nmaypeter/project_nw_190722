from SeedSelection import *
from Evaluation import *
import time
import copy


class Model:
    def __init__(self, model_name, dataset_name, product_name, cascade_model, wallet_distribution_type=''):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.budget_iteration = [i for i in range(10, 5, -1)]
        self.wallet_distribution_type = wallet_distribution_type
        self.wd_seq = ['m50e25', 'm99e96']
        self.monte_carlo = 100

    def model_mioa(self, epw_flag):
        ini = Initialization(self.dataset_name, self.product_name)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        num_node = getTotalNumNode(graph_dict) * num_product

        seed_set_sequence = []
        ss_time_sequence = []
        ssmioa_model = SeedSelectionMIOA(graph_dict, product_list, product_weight_list)

        ss_start_time = time.time()
        bud_iter = self.budget_iteration.copy()
        b_iter = bud_iter.pop(0)
        total_budget = max(int(safe_div(num_node, 2 ** b_iter)), 1)

        now_budget, now_profit = 0, 0.0
        seed_set = [set() for _ in range(num_product)]
        seed_mioa_dict = [{} for _ in range(num_product)]
        wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
        mioa_dict = ssmioa_model.generateMIOA()
        if epw_flag:
            mioa_dict = ssmioa_model.updateMIOAEPW(mioa_dict)
            celf_heap = [(sum(mioa_dict[k][i][j][0] for j in mioa_dict[k][i]) * product_list[k][0], k, i, 0) for k in range(num_product) for i in mioa_dict[k]]
        else:
            celf_heap = [(sum(mioa_dict[i][j][0] for j in mioa_dict[i]) * product_list[k][0] * product_weight_list[k], k, i, 0) for k in range(num_product) for i in mioa_dict]
        heap.heapify_max(celf_heap)

        seed_data = []
        print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
              ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget))

        while now_budget < total_budget and celf_heap:
            mep_item = heap.heappop_max(celf_heap)
            mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            if mep_flag == now_budget:
                if epw_flag:
                    seed_mioa_dict = updateSeedMIOADict(seed_mioa_dict, mep_k_prod, mep_i_node, seed_set, mioa_dict[mep_k_prod][mep_i_node])
                else:
                    seed_mioa_dict = updateSeedMIOADict(seed_mioa_dict, mep_k_prod, mep_i_node, seed_set, mioa_dict[mep_i_node])
                seed_set[mep_k_prod].add(mep_i_node)
                now_budget += 1
                now_profit = round(now_profit + mep_mg, 4)
                seed_data.append(str(round(time.time() - ss_start_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                 str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')
            else:
                if epw_flag:
                    seed_exp_mioa_dict = updateSeedMIOADict(copy.deepcopy(seed_mioa_dict), mep_k_prod, mep_i_node, seed_set, mioa_dict[mep_k_prod][mep_i_node])
                else:
                    seed_exp_mioa_dict = updateSeedMIOADict(copy.deepcopy(seed_mioa_dict), mep_k_prod, mep_i_node, seed_set, mioa_dict[mep_i_node])
                expected_inf = calculateExpectedInf(seed_exp_mioa_dict)
                ep_t = sum(expected_inf[k] * product_list[k][0] for k in range(num_product))
                mg_t = round(ep_t - now_profit, 4)
                flag_t = now_budget

                if mg_t > 0:
                    celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                    heap.heappush_max(celf_heap, celf_item_t)

            if now_budget == total_budget:
                ss_time = round(time.time() - ss_start_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence.append(copy.deepcopy(seed_set))
                ss_time_sequence.append(ss_time)

                if bud_iter:
                    b_iter = bud_iter.pop(0)
                    total_budget = max(int(safe_div(num_node, 2 ** b_iter)), 1)

        for wd in wd_seq:
            seed_data_path = 'seed_data/' + self.model_name + '_' + wd + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '.txt'
            seed_data_file = open(seed_data_path, 'w')
            for sd in seed_data:
                seed_data_file.write(sd)
            seed_data_file.close()

        while len(seed_set_sequence) != len(self.budget_iteration):
            seed_set_sequence.append(seed_set_sequence[-1])
            ss_time_sequence.append(ss_time_sequence[-1])

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_dag(self, dag_pointer):
        ini = Initialization(self.dataset_name, self.product_name)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        num_node = getTotalNumNode(graph_dict) * num_product

        seed_set_sequence = []
        ss_time_sequence = []
        ssmioa_model = SeedSelectionMIOA(graph_dict, product_list, product_weight_list)

        ss_start_time = time.time()
        bud_iter = self.budget_iteration.copy()
        b_iter = bud_iter.pop(0)
        total_budget = max(int(safe_div(num_node, 2 ** b_iter)), 1)

        now_budget, now_profit = 0, 0.0
        seed_set = [set() for _ in range(num_product)]
        seed_dag_dict = [{} for _ in range(num_product)]
        wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
        mioa_dict = ssmioa_model.generateMIOA()
        celf_heap = [(sum(mioa_dict[i][j][0] for j in mioa_dict[i]) * product_list[k][0] * product_weight_list[k], k, i, 0) for k in range(num_product) for i in mioa_dict]
        heap.heapify_max(celf_heap)
        mep_seed_dag_dict = (celf_heap[0][0], [{} for _ in range(num_product)])
        mep_seed_dag_dict[1][celf_heap[0][1]] = {celf_heap[0][2]: mioa_dict[celf_heap[0][2]]}

        seed_data = []
        print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
              ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget))

        while now_budget < total_budget and celf_heap:
            mep_item = heap.heappop_max(celf_heap)
            mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            if mep_flag == now_budget:
                seed_set[mep_k_prod].add(mep_i_node)
                now_budget += 1
                now_profit = round(now_profit + mep_mg, 4)
                seed_dag_dict = mep_seed_dag_dict[1]
                mep_seed_dag_dict = (0.0, [{} for _ in range(num_product)])
                seed_data.append(str(round(time.time() - ss_start_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                 str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')
            else:
                seed_exp_dag_dict = copy.deepcopy(seed_dag_dict)
                updateSeedDAGDict(seed_exp_dag_dict, mep_k_prod, mep_i_node)
                seed_set_t = seed_set[mep_k_prod].copy()
                seed_set_t.add(mep_i_node)
                dag_k_dict = ssmioa_model.generateDAG1(seed_set_t) if dag_pointer == 1 else ssmioa_model.generateDAG2(seed_set_t, {i: mioa_dict[i] for i in seed_set_t})
                seed_exp_dag_dict[mep_k_prod] = ssmioa_model.generateSeedDAGDict(dag_k_dict, seed_set_t)
                expected_inf = calculateExpectedInf(seed_exp_dag_dict)
                ep_t = sum(expected_inf[k] * product_list[k][0] * product_weight_list[k] for k in range(num_product))
                mg_t = round(ep_t - now_profit, 4)
                flag_t = now_budget

                if mg_t > 0:
                    celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                    heap.heappush_max(celf_heap, celf_item_t)
                    if mg_t > mep_seed_dag_dict[0]:
                        mep_seed_dag_dict = (mg_t, seed_exp_dag_dict)

            if now_budget == total_budget:
                ss_time = round(time.time() - ss_start_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence.append(copy.deepcopy(seed_set))
                ss_time_sequence.append(ss_time)

                if bud_iter:
                    b_iter = bud_iter.pop(0)
                    total_budget = max(int(safe_div(num_node, 2 ** b_iter)), 1)

        for wd in wd_seq:
            seed_data_path = 'seed_data/' + self.model_name + '_' + wd + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '.txt'
            seed_data_file = open(seed_data_path, 'w')
            for sd in seed_data:
                seed_data_file.write(sd)
            seed_data_file.close()

        while len(seed_set_sequence) != len(self.budget_iteration):
            seed_set_sequence.append(seed_set_sequence[-1])
            ss_time_sequence.append(ss_time_sequence[-1])

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ng(self):
        ini = Initialization(self.dataset_name, self.product_name)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        num_node = getTotalNumNode(graph_dict) * num_product

        seed_set_sequence = []
        ss_time_sequence = []
        ssng_model = SeedSelectionNG(graph_dict, product_list, product_weight_list)
        diff_model = Diffusion(graph_dict, product_list, product_weight_list)

        ss_start_time = time.time()
        bud_iter = self.budget_iteration.copy()
        b_iter = bud_iter.pop(0)
        total_budget = max(int(safe_div(num_node, 2 ** b_iter)), 1)

        now_budget, now_profit = 0, 0.0
        seed_set = [set() for _ in range(num_product)]
        wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
        celf_heap = ssng_model.generateCelfHeap()

        seed_data = []
        print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
              ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget))

        while now_budget < total_budget and celf_heap:
            mep_item = heap.heappop_max(celf_heap)
            mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            if mep_flag == now_budget:
                seed_set[mep_k_prod].add(mep_i_node)
                now_budget += 1
                now_profit = safe_div(sum([diff_model.getSeedSetProfit(seed_set) for _ in range(self.monte_carlo)]), self.monte_carlo)
                seed_data.append(str(round(time.time() - ss_start_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                 str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')
            else:
                seed_set_t = copy.deepcopy(seed_set)
                seed_set_t[mep_k_prod].add(mep_i_node)
                ep_t = safe_div(sum([diff_model.getSeedSetProfit(seed_set_t) for _ in range(self.monte_carlo)]), self.monte_carlo)
                mg_t = round(ep_t - now_profit, 4)
                flag_t = now_budget

                if mg_t > 0:
                    celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                    heap.heappush_max(celf_heap, celf_item_t)

            if now_budget == total_budget:
                ss_time = round(time.time() - ss_start_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence.append(copy.deepcopy(seed_set))
                ss_time_sequence.append(ss_time)

                if bud_iter:
                    b_iter = bud_iter.pop(0)
                    total_budget = max(int(safe_div(num_node, 2 ** b_iter)), 1)

        for wd in wd_seq:
            seed_data_path = 'seed_data/' + self.model_name + '_' + wd + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '.txt'
            seed_data_file = open(seed_data_path, 'w')
            for sd in seed_data:
                seed_data_file.write(sd)
            seed_data_file.close()

        while len(seed_set_sequence) != len(self.budget_iteration):
            seed_set_sequence.append(seed_set_sequence[-1])
            ss_time_sequence.append(ss_time_sequence[-1])

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_hd(self):
        ini = Initialization(self.dataset_name, self.product_name)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        num_node = getTotalNumNode(graph_dict) * num_product

        seed_set_sequence = []
        ss_time_sequence = []
        sshd_model = SeedSelectionHD(graph_dict, product_list)

        ss_start_time = time.time()
        bud_iter = self.budget_iteration.copy()
        b_iter = bud_iter.pop(0)
        total_budget = max(int(safe_div(num_node, 2 ** b_iter)), 1)

        now_budget, now_profit = 0, 0.0
        seed_set = [set() for _ in range(num_product)]
        wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
        degree_heap = sshd_model.generateDegreeHeap()

        seed_data = []
        print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
              ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget))

        while now_budget < total_budget and degree_heap:
            mep_item = heap.heappop_max(degree_heap)
            mep_deg, mep_k_prod, mep_i_node = mep_item

            seed_set[mep_k_prod].add(mep_i_node)
            now_budget += 1

            if now_budget == total_budget:
                ss_time = round(time.time() - ss_start_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence.append(copy.deepcopy(seed_set))
                ss_time_sequence.append(ss_time)

                if bud_iter:
                    b_iter = bud_iter.pop(0)
                    total_budget = max(int(safe_div(num_node, 2 ** b_iter)), 1)

        for wd in wd_seq:
            seed_data_path = 'seed_data/' + self.model_name + '_' + wd + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '.txt'
            seed_data_file = open(seed_data_path, 'w')
            for sd in seed_data:
                seed_data_file.write(sd)
            seed_data_file.close()

        while len(seed_set_sequence) != len(self.budget_iteration):
            seed_set_sequence.append(seed_set_sequence[-1])
            ss_time_sequence.append(ss_time_sequence[-1])

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_r(self):
        ini = Initialization(self.dataset_name, self.product_name)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        num_node = getTotalNumNode(graph_dict) * num_product

        seed_set_sequence = []
        ss_time_sequence = []
        ssr_model = SeedSelectionRandom(graph_dict, product_list)

        ss_start_time = time.time()
        bud_iter = self.budget_iteration.copy()
        b_iter = bud_iter.pop(0)
        total_budget = max(int(safe_div(num_node, 2 ** b_iter)), 1)

        now_budget, now_profit = 0, 0.0
        seed_set = [set() for _ in range(num_product)]
        wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
        random_node_list = ssr_model.generateRandomList()

        seed_data = []
        print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
              ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget))

        while now_budget < total_budget and random_node_list:
            mep_item = random_node_list.pop(0)
            mep_k_prod, mep_i_node = mep_item

            seed_set[mep_k_prod].add(mep_i_node)
            now_budget += 1

            if now_budget == total_budget:
                ss_time = round(time.time() - ss_start_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence.append(copy.deepcopy(seed_set))
                ss_time_sequence.append(ss_time)

                if bud_iter:
                    b_iter = bud_iter.pop(0)
                    total_budget = max(int(safe_div(num_node, 2 ** b_iter)), 1)

        for wd in wd_seq:
            seed_data_path = 'seed_data/' + self.model_name + '_' + wd + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '.txt'
            seed_data_file = open(seed_data_path, 'w')
            for sd in seed_data:
                seed_data_file.write(sd)
            seed_data_file.close()

        while len(seed_set_sequence) != len(self.budget_iteration):
            seed_set_sequence.append(seed_set_sequence[-1])
            ss_time_sequence.append(ss_time_sequence[-1])

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])