# import _init_paths
# from core.evaluate import balanced_f1
# from sklearn.metrics import balanced_accuracy_score, f1_score

# if __name__ == "__main__":
# 	y_true = [0, 0, 0, 1, 2, 2, 2, 2, 2, 2]
# 	y_pred = [0, 0, 1, 2, 0, 1, 1, 2, 2, 2]
# 	# sample_weight = [1/3, 1/3, 1/3, 0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
# 	# sample_weight = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
# 	print(balanced_f1(y_pred=y_pred, y_true=y_true, per_class=True))


import _init_paths
import click
from loss import *
from config import config, SimpleConfig
from utils.utils import (
    deep_update_dict,
    create_logger,
    get_optimizer,
    get_scheduler,
    get_model,
    get_category_list,
    get_dataset,
)
from tensorboardX import SummaryWriter
from core.function import test_model
import torch
import os
import json
import utils
from os import path as osp
import shutil
from torch.utils.data import DataLoader
import argparse
import warnings
from functools import partial
from datetime import datetime
from dataset.base_dataset import Dataset, dgl_collate_func, mpnn_collate_func, default_collate_func

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    apex_available = True
except:
    print('*-*'*30)
    print('Apex have not been installed! Code will be run without using apex!')
    print('*-*'*30)
    apex_available = False
    use_apex = False
    pass

def test():
    utils.global_seed(cfg['seed'])
    print('cfg',cfg)
    device = utils.set_gpu_mode(cfg['use_gpu'], cfg['gpu_id'])
    print('device', device)
    local_rank = cfg['train']['local_rank']
    rank = local_rank
    logger, log_file, exp_id = create_logger(cfg, local_rank, test=True)
    warnings.filterwarnings("ignore")

    # close loop 
    model_dir = osp.join(cfg['output_dir'], cfg['dataset']['dataset_name'], "models", cfg['test']['exp_id'])
    # code_dir = osp.join(cfg['output_dir'], cfg['dataset']['dataset_name'], "code")


    # ----- BEGIN DATASET BUILDER -----
    datasets = get_dataset(cfg, test=True)
    test_set = datasets['test_set']
    print('dataset', datasets)
    entity_type = test_set.entity_type

    # ----- END DATASET BUILDER -----

    if cfg['setting']['type'] not in ['LT Regression', 'LT Generation']:
        num_class_list = get_category_list(test_set)
        num_classes = len(num_class_list) - 1 # the model was trained only with closes sets, without the outlier class in the open set
        para_dict = {
            "num_classes": num_classes,
            "num_class_list": num_class_list,
            "cfg": cfg,
            "device": device,
        }
        cfg['setting']['num_class'] = num_classes # update the real number of classes based on the datasets

    # ----- BEGIN MODEL BUILDER -----
    model = get_model(cfg=cfg, device=device, logger=logger, entity_type=entity_type)
    model_file = os.path.join(model_dir, cfg['test']['model_file'])
    model.load_model(model_file)
    model = torch.nn.DataParallel(model).cuda()

    # ----- END MODEL BUILDER -----
    params = {}
    if (cfg['dataset']['drug_encoding'] == "MPNN"):
        params['collate_fn'] = partial(mpnn_collate_func, entity_type=entity_type)
    elif cfg['dataset']['drug_encoding'] in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', \
        'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
        params['collate_fn'] = partial(dgl_collate_func, entity_type=entity_type)
    else:
        params['collate_fn'] = partial(default_collate_func, entity_type=entity_type)


    testLoader = DataLoader(
        test_set,
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=cfg['test']['num_workers'],
        pin_memory=False,
        drop_last=False,
        **params
    )

    logger.info(
        "-------------------Test start :{}  {}  {}-------------------".format(
            cfg['dataset'], cfg['neck']['type'], cfg['train']['combiner']['type']
        )
    )

    test_model(testLoader, model, cfg, logger, device)

if __name__ == "__main__":

    path = "../configs/single_pred/LT_Regression/baseline/QM9.json"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=path)
    args = parser.parse_args()
    cfg = config

    # import importlib.util
    # spec = importlib.util.spec_from_file_location("module.name", args.config)
    # exp_params = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(exp_params)
    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    test()

