from Model import *

if __name__ == '__main__':
    dataset_seq = [1]
    prod_seq = [1]
    cm_seq = [1]
    wd_seq = [1]

    for data_setting in dataset_seq:
        dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                       'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            for prod_setting in prod_seq:
                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)

                Model('mmioa', dataset_name, product_name, cascade_model).model_mioa(epw_flag=False)
                Model('mdag1', dataset_name, product_name, cascade_model).model_dag(dag_pointer=1)
                Model('mdag2', dataset_name, product_name, cascade_model).model_dag(dag_pointer=2)
                Model('mng', dataset_name, product_name, cascade_model).model_ng()
                Model('mhd', dataset_name, product_name, cascade_model).model_hd()
                Model('mr', dataset_name, product_name, cascade_model).model_r()
                # Model('mpmis', dataset_name, product_name, cascade_model).model_pmis()

                for wd in wd_seq:
                    wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2)

                    Model('mmioaepw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_mioa(epw_flag=True)
                    Model('mmioapw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_mioa(epw_flag=False)
                    Model('mdag1pw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_dag(dag_pointer=1)
                    Model('mdag2pw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_dag(dag_pointer=2)
                    Model('mngpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_ng()