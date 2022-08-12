import numpy as np
import pandas as pd
import os
import json
import subprocess
import argparse



def init_df():
    df_list = []

    all_metrics = [metric if i == 0 else metric+'_'+lts[i-1] for metric in metrics for i in range(len(lts)+1)]
    col_index = [(dataset, metric) for dataset in datasets for metric in all_metrics]
    row_index = [(, backbone) for  in s for backbone in backbones]

    col_index = pd.MultiIndex.from_tuples(col_index)
    row_index = pd.MultiIndex.from_tuples(row_index)

    for seed in seeds:
        df = pd.DataFrame(data=np.nan, columns=col_index, index=row_index)
        df_list.append(df)

    return df_list


def get_log(path):
    log_list = []
    g = os.walk(path)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            log_list.append(os.path.join(path, file_name))
    return log_list

def get_params(log):
    with open(log, 'r') as f:
        lines = f.readlines()
        for line in lines:
            idx = line.find('{')
            if idx != -1:
                params = eval(line[idx:])
                assert isinstance(params, dict)

                return params

def get_result(log):
    result = dict()
    with open(log, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.find('{') != -1:
                continue
            for metric in metrics:
                idx = line.find(metric)
                if idx != -1:
                    result[metric] = eval(line[idx+len(metric)+1:idx+len(metric)+8])
                    for lt in lts:
                        entry = metric+'_'+lt
                        idx = line.find(entry)
                        try:
                            result[entry] = eval(line[idx+len(entry)+1:idx+len(entry)+7])
                        except:
                            pass

    return result

metrics = ['balanced_accuracy', 'balanced-f1', 'roc-auc']
lts = ['head', 'middle', 'tail']
datasets = ['uspto_500_MT']
s = ['CrossEntropy', 'DIVE', 'CDT', 'Decoupling', 'IBLoss', 'remix', 'mix_up', 'bbn_mix',
             'BalancedSoftmaxCE', 'ClassBalanceFocal', 'CostSensitiveCE']
backbones = ['Morgan', 'DGL_GCN', 'Transformer']
seeds = [0, 1, 2]


if __name__ == '__main__':

    setting_type = 'LT Regression'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_path',
        type=str,
        default="/apdcephfs/share_1364275/shenyuan/DrugLT/500MT_time_result")
    parser.add_argument(
        '--result_path',
        type=str,
        default="/apdcephfs/share_1364275/shenyuan/DrugLT/500MT_time_result.xlsx")
    args = parser.parse_args()

    if setting_type == 'LT Regression':
        metrics = ['mse', 'mae']
        lts = []
        s = ['MSE', 'FocalR', 'BalancedMSELoss', 'remix', 'mix_up', 'bbn_mix', 'LDS']
    elif setting_type == 'LT Classification':
        metrics = ['balanced_accuracy', 'balanced-f1', 'roc-auc']
        lts = ['head', 'middle', 'tail']
        s = ['CrossEntropy', 'DIVE', 'CDT', 'Decoupling', 'IBLoss', 'remix', 'mix_up', 'bbn_mix',
                     'BalancedSoftmaxCE', 'ClassBalanceFocal', 'CostSensitiveCE']


    df_list = init_df()

    log_path = args.log_path
    result_path = args.result_path

    log_list = get_log(log_path)

    for log in log_list:
        params = get_params(log)



        result = get_result(log)
        if not result:
            continue

        dataset = params['dataset']['dataset_name']
        try:
             = params['_name']
        except:
            continue
        backbone = params['dataset']['drug_encoding']
        seed = params['seed']
        split = params['dataset']['split']['method']

        for metric, value in result.items():
            df_list[seed][dataset, metric][, backbone] = value

    writer = pd.ExcelWriter(result_path)
    for i, df in enumerate(df_list):
        df.to_excel(writer, sheet_name=split+str(i))
    writer.save()




    