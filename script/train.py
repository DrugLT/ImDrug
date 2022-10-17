
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
    set_baseline,
    get_dataset,
    set_distributed,
    set_resume
)
from tensorboardX import SummaryWriter
from core.function import train_model, valid_model
from core.combiner import Combiner
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

def train():
    utils.global_seed(cfg['seed']) # set global seed for numpy, random, pytorch, dgl, cuda and cudnn 
    print('cfg',cfg)
    device = utils.set_gpu_mode(cfg['use_gpu'], cfg['gpu_id'])
    print('device', device)
    local_rank = cfg['train']['local_rank']
    rank = local_rank # for distributed training
    logger, log_file, exp_id = create_logger(cfg, local_rank)
    warnings.filterwarnings("ignore")

    auto_resume = cfg['train']['auto_resume'] # for resumed training

    # close loop 
    model_dir = osp.join(cfg['output_dir'], cfg['dataset']['dataset_name'], "models", exp_id)
    
    tensorboard_dir = (
        os.path.join(cfg['output_dir'], cfg['name'], "tensorboard")
        if cfg['train']['tensorboard']['enable']
        else None
    )

    set_distributed(cfg=cfg, local_rank=local_rank, model_dir=model_dir, tensorboard_dir=tensorboard_dir)

    # ----- BEGIN DATASET BUILDER -----
    datasets = get_dataset(cfg)
    train_set = datasets['train_set']
    valid_set = datasets['valid_set']
    entity_type = train_set.entity_type

    # ----- END DATASET BUILDER -----

    # ----- BEGIN LOSS BUILDER -----
    if cfg['setting']['type'] not in ['LT Regression', 'LT Generation']:
        num_class_list = get_category_list(train_set)
        num_classes = len(num_class_list)
        para_dict = {
            "num_classes": num_classes,
            "num_class_list": num_class_list,
            "cfg": cfg,
            "device": device,
        }
        cfg['setting']['num_class'] = num_classes # update the real number of classes based on the datasets

        criterion = eval(cfg['loss']['type'])(para_dict)
    elif cfg['setting']['type'] == 'LT Regression':
        para_dict = {
            "cfg": cfg,
            "device": device,
        }
        criterion = eval(cfg['loss']['type'])(para_dict)
        num_class_list = None
    else:
        raise NotImplementedError
    epoch_number = cfg['train']['max_epoch']

    # ----- END LOSS BUILDER -----

    # ----- BEGIN MODEL BUILDER -----
    model = get_model(cfg=cfg, device=device, logger=logger, entity_type=entity_type)
    combiner = Combiner(cfg, device, num_class_list)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    # ----- END MODEL BUILDER -----

    # ----- BEGIN DATALOADER BUILDER -----
    params = {}
    if (cfg['dataset']['drug_encoding'] == "MPNN"):
        params['collate_fn'] = partial(mpnn_collate_func, entity_type=entity_type)
    elif cfg['dataset']['drug_encoding'] in ['DGL_GCN', 'DGL_NeuralFP', \
        'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
        params['collate_fn'] = partial(dgl_collate_func, entity_type=entity_type)
    else:
        params['collate_fn'] = partial(default_collate_func, entity_type=entity_type)

    if cfg['train']['distributed']:
        train_sampler = torch.utils.data.DistributedSampler(train_set)
        val_sampler = torch.utils.data.DistributedSampler(valid_set)
        trainLoader = DataLoader(
            train_set,
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
            num_workers=cfg['train']['num_workers'],
            pin_memory=False,
            sampler=train_sampler,
            **params
        )
        validLoader = DataLoader(
            valid_set,
            batch_size=cfg['test']['batch_size'],
            shuffle=False,
            num_workers=cfg['test']['num_workers'],
            pin_memory=False,
            sampler=val_sampler,
            **params
        )

    else:
        trainLoader = DataLoader(
            train_set,
            batch_size=cfg['train']['batch_size'],
            shuffle=cfg['train']['shuffle'],
            num_workers=cfg['train']['num_workers'],
            pin_memory=cfg['pin_memory'],
            drop_last=True,
            **params
        )

        validLoader = DataLoader(
            valid_set,
            batch_size=cfg['test']['batch_size'],
            shuffle=False,
            num_workers=cfg['test']['num_workers'],
            pin_memory=cfg['pin_memory'],
            **params
        )
    
    # ----- END DATALOADER BUILDER

    if tensorboard_dir is not None and local_rank == 0:
        writer = SummaryWriter(log_dir=tensorboard_dir)
    else:
        writer = None

    if cfg['setting']['type'] == "LT Regression":
        best_result, best_epoch, start_epoch = float('inf'), 0, 1
    else:
        best_result, best_epoch, start_epoch = 0, 0, 1

    # ----- BEGIN RESUME ---------
    optimizer, scheduler, start_epoch, best_result, best_epoch = set_resume(cfg, optimizer, \
        scheduler, start_epoch, best_result, best_epoch, model_dir, auto_resume)

    # ----- END RESUME ---------

    if local_rank == 0:
        logger.info(
            "-------------------Train start :{}  {}  {}-------------------".format(
                cfg['dataset'], cfg['neck']['type'], cfg['train']['combiner']['type']
            )
        )

    train_acc_list = []
    val_acc_list = []

    # ----- BEGIN TRAINING ITERATION ---------
    for epoch in range(start_epoch, epoch_number + 1):
        if cfg['train']['distributed']:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        scheduler.step()
        train_acc, train_loss = train_model(
            trainLoader,
            model,
            epoch,
            epoch_number,
            optimizer,
            combiner,
            criterion,
            cfg,
            logger,
            writer=writer,
            rank=local_rank,
            use_apex=use_apex
        )
        model_save_path = os.path.join(
            model_dir,
            "epoch_{}.pth".format(epoch),
        )
        if epoch % cfg['save_step'] == 0 and local_rank == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_result': best_result,
                'best_epoch': best_epoch,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict() if use_apex else {}
            }, model_save_path)

        loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}
        if cfg['valid_step'] != -1 and epoch % cfg['valid_step'] == 0:
            valid_acc, valid_loss, valid_results, valid_metrics = valid_model(
                validLoader, epoch, model, cfg, criterion, logger, device,
                rank=rank, distributed=cfg['train']['distributed'], writer=writer
            )
            train_acc_list.append(train_acc)
            val_acc_list.append(valid_acc)
            loss_dict["valid_loss"], acc_dict["valid_acc"] = valid_loss, valid_acc
            if cfg['setting']['type'] == "LT Regression":
                if valid_results[0] < best_result and local_rank == 0:
                    best_result, best_epoch = valid_results[0], epoch
                    torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_result': best_result,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict() if use_apex else {}
                    }, os.path.join(model_dir, "best_model.pth")
                    )

            elif valid_results[0] > best_result and local_rank == 0:
                best_result, best_epoch = valid_results[0], epoch
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_result': best_result,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict() if use_apex else {}
                }, os.path.join(model_dir, "best_model.pth")
                )
            if rank == 0:
                logger.info(
                    "--------------Best_Epoch:{:>3d}    Best_{}:{:>5.2f}%--------------".format(
                        best_epoch, valid_metrics[0], best_result * 100
                    )
                )

        if cfg['train']['tensorboard']['enable'] and local_rank == 0:
            writer.add_scalars("scalar/acc", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)
    if cfg['train']['tensorboard']['enable'] and local_rank == 0:
        writer.close()
    if rank == 0:
        logger.info(
            "-------------------Train Finished :{}-------------------".format(cfg['name'])
        )
    
    # ----- END TRAINING ITERATION ---------

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=None)

    args = parser.parse_args()
    cfg = config

    if args.config is not None:
        with open(args.config, "r") as f:
            exp_params = json.load(f)
    else:
        exp_params = {'baseline': 'Default_cls'}

    cfg = set_baseline(exp_params, cfg)
    cfg = deep_update_dict(exp_params, cfg)

    train()
