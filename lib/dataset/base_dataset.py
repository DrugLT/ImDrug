# -*- coding: utf-8 -*-
# Author: ImDrug Team
# License: MIT 
"""
This file contains a base data loader object that specific one can inherit from. 
"""

import pandas as pd
import numpy as np
import sys
import warnings
import dgl
import torch
import random
import math
from typing import List
import DeepPurpose.utils
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang

warnings.filterwarnings("ignore")

import utils

###########################
# collate fn for dataloader
###########################

class DeviceList(list):

    def __init__(self, *args):
        super(DeviceList, self).__init__(*args)

    def to(self, device):
        dl = DeviceList()
        for i in self:
            dl.append(i.to(device))
        return dl

def default_collate_func(batch, entity_type):
    elem = batch[0]
    dic = {}
    for key, _ in elem.items():
        dic[key] = default_collate_helper([d[key] for d in batch], key, entity_type) 
    return dic

def dgl_collate_func(batch, entity_type):
    # delete None in batch
    batch = [d for d in batch if None not in d['x']] 
    
    elem = batch[0]
    dic = {}
    for key, _ in elem.items():
        if key == 'meta':
            dic[key] = None
        else:
            dic[key] = dgl_collate_helper([d[key] for d in batch], key, entity_type) 
    return dic

def mpnn_collate_func(batch, entity_type):
    elem = batch[0]
    dic = {}
    for key, _ in elem.items():
        dic[key] = mpnn_collate_helper([d[key] for d in batch], key, entity_type) 
    return dic

def dgl_collate_helper(elems, key, entity_type):
    if key == "meta":
        if len(elems[0]) == 0:
            return None
    if key == "meta_data" or key == 'meta_label':
        if elems[0] == None:
            return None
    if key in ["x", "meta_data"]:
        data_lst = list(zip(*elems))
        for i, entity in enumerate(entity_type):
            if entity == 'drug':
                data_lst[i] = dgl.batch(data_lst[i])
            else:
                data_lst[i] = torch.tensor(data_lst[i]).float()      

        return DeviceList(data_lst)
    else:
        return torch.utils.data.dataloader.default_collate(elems)

def default_collate_helper(elems, key, entity_type):
    if key == "meta":
        if len(elems[0]) == 0:
            return None
    if key == "meta_data" or key == 'meta_label':
        if elems[0] == None:
            return None
    if key in ["x", "meta_data"]:
        data_lst = list(zip(*elems))
        for i, entity in enumerate(entity_type):
            data_lst[i] = torch.tensor(data_lst[i]).float()      

        return DeviceList(data_lst)
    else:
        return torch.utils.data.dataloader.default_collate(elems)

def mpnn_collate_helper(elems, key, entity_type):
    if key == "meta":
        if len(elems[0]) == 0:
            return None
    if key == "meta_data" or key == 'meta_label':
        if elems[0] == None:
            return None
    if key in ["x", "meta_data"]:
        data_lst = list(zip(*elems))
        for i, entity in enumerate(entity_type):
            if entity == 'drug':
                data = data_lst[i]
                mpnn_feature = [j[0] for j in data]
                mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
                from torch.utils.data.dataloader import default_collate
                x_remain = [list(j[1:]) for j in data]
                x_remain_collated = default_collate(x_remain)
                data_lst[i] = [mpnn_feature] + x_remain_collated
            else:
                data_lst[i] = torch.tensor(data_lst[i]).float()  
        return DeviceList(data_lst)
    else:
        return torch.utils.data.dataloader.default_collate(elems) 

def mpnn_feature_collate_func(x):
    N_atoms_scope = torch.cat([i[4] for i in x], 0)
    f_a = torch.cat([x[j][0].unsqueeze(0) for j in range(len(x))], 0)
    f_b = torch.cat([x[j][1].unsqueeze(0) for j in range(len(x))], 0)
    agraph_lst, bgraph_lst = [], []
    for j in range(len(x)):
        agraph_lst.append(x[j][2].unsqueeze(0))
        bgraph_lst.append(x[j][3].unsqueeze(0))
    agraph = torch.cat(agraph_lst, 0)
    bgraph = torch.cat(bgraph_lst, 0)
    return [f_a, f_b, agraph, bgraph, N_atoms_scope]


def get_lds_kernel_window(kernel, ks, sigma):
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, entities: List[str], entity_type: List[str], 
                 data_df: pd.DataFrame,
                #  label_name: str = 'Y', label_weight_name: str = 'Y_Weight',
                #  task='single_pred', drug_encoding='DGL_GCN',
                #  protein_encoding='Transformer', 
                 mode='train'
                 ):
        self.cfg = cfg
        self.task = cfg['dataset']['tier1_task']
        self.mode = mode
        self.data_df = data_df
        self.label_name = cfg['dataset']['split']['label_name']
        self.label_weight_name = cfg['dataset']['split']['label_weight_name']
        self.lt_label_name = cfg['dataset']['split']['lt_label_name']
        self.entities = entities
        self.entity_type = entity_type
        self.drug_encoding = cfg['dataset']['drug_encoding']
        self.protein_encoding = cfg['dataset']['protein_encoding']

        if self.mode == "train":
            print("Loading train data ...", end=" ")
        elif "valid" in self.mode:
            print("Loading valid data ...", end=" ")
        elif "test" in self.mode:
            print("Loading test data ...", end=" ")            
        else:
            raise NotImplementedError

        if self.drug_encoding in ['DGL_GCN', 'DGL_NeuralFP']:
            from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
            self.node_featurizer = CanonicalAtomFeaturizer()
            self.edge_featurizer = CanonicalBondFeaturizer(self_loop = True)
            from functools import partial
            self.fc = partial(smiles_to_bigraph, node_featurizer = self.node_featurizer, edge_featurizer = self.edge_featurizer, add_self_loop=True)
        elif self.drug_encoding == 'DGL_AttentiveFP':
            from dgllife.utils import smiles_to_bigraph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
            self.node_featurizer = AttentiveFPAtomFeaturizer()
            self.edge_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
            from functools import partial
            self.fc = partial(smiles_to_bigraph, node_featurizer = self.node_featurizer, edge_featurizer = self.edge_featurizer, add_self_loop=True)
        elif self.drug_encoding in ['DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred']:
            from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
            self.node_featurizer = PretrainAtomFeaturizer()
            self.edge_featurizer = PretrainBondFeaturizer()
            from functools import partial
            self.fc = partial(smiles_to_bigraph, node_featurizer = self.node_featurizer, edge_featurizer = self.edge_featurizer, add_self_loop=True)

        self.featurize()
        self.load()

        if self.cfg['setting'] not in ['LT Regression', 'LT Generation']:
            self.class_weight, self.sum_weight = self.get_weight()
            self.class_dict = self._get_class_dict()

            if (cfg['train']['sampler']['type']  == "weighted_sampler" or cfg['train']['sampler']['type']  == "bbn_sampler") and self.mode == "train":
                class_label = np.array(self.data[self.label_name])
                self.num_class = len(set(class_label))
                num_list = []
                for i in range(self.num_class):
                    num_list.append(np.sum(class_label == i))

                self.instance_p = np.array([num / sum(num_list) for num in num_list])
                self.class_p = np.array([1/self.num_class for _ in num_list])
                num_list = [math.sqrt(num) for num in num_list]
                self.square_p = np.array([num / sum(num_list) for num in num_list])
                # self.class_dict = self._get_class_dict()

        if self.cfg['loss']['type'] == 'LDS':
            self.lds_weights = self.get_lds_weights(reweight=self.cfg['loss']['LDS']['reweight'],
                                                    lds_kernel=self.cfg['loss']['LDS']['kernel'],
                                                    lds_sigma=self.cfg['loss']['LDS']['SIGMA'],
                                                    lds_ks=self.cfg['loss']['LDS']['ks'])
    
    def load(self):
        self.data = {}
        self.data['entities'] = self.entities
        self.data['entity_type'] = self.entity_type
        self.data[self.label_name] = self.feat_df[self.label_name].to_list()
        if self.label_weight_name not in self.feat_df.columns:
            self.feat_df[self.label_weight_name] = 1 # set sample weight to 1 by default, if not specified during data splitting
        if self.lt_label_name not in self.feat_df.columns:
            self.feat_df[self.lt_label_name] = self.feat_df[self.label_name] # set lt_label_name to label_name by default, if not specified otherwise
        self.data[self.label_weight_name] = self.feat_df[self.label_weight_name].to_list()
        self.data[self.lt_label_name] = self.feat_df[self.lt_label_name].to_list()
        if self.cfg['setting'] not in ['LT Regression', 'LT Generation']:
            self.num_class = len(set(self.data[self.label_name]))
        for entity in self.entities:
            self.data[entity] = self.feat_df[entity].to_list()

    def featurize(self):
        for entity, entity_type in zip(self.entities, self.entity_type):
            if entity_type == 'drug':
                self.feat_df = DeepPurpose.utils.encode_drug(df_data=self.data_df, 
                                                             drug_encoding=self.drug_encoding, 
                                                             column_name=entity,
                                                             save_column_name=entity)
                if self.drug_encoding == 'CNN' or self.drug_encoding == 'CNN_RNN':
                    self.feat_df[entity] = self.feat_df[entity].apply(DeepPupose.utils.drug_2_embed)
                elif self.drug_encoding in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
                    self.feat_df[entity] = self.feat_df[entity].apply(self.fc)
                

            elif entity_type == 'protein':
                self.feat_df = DeepPurpose.utils.encode_protein(df_data=self.data_df, 
                                                                target_encoding=self.protein_encoding, 
                                                                column_name=entity,
                                                                save_column_name=entity)
                if self.protein_encoding == 'CNN' or self.protein_encoding == 'CNN_RNN':
                    self.feat_df[entity] = self.feat_df[entity].apply(DeepPurpose.utils.protein_2_embed)

            else:
                raise NotImplementedError

    def __len__(self):
        return len(self.data[self.entities[0]])

    def __getitem__(self, index):
        if self.cfg['train']['sampler']['type'] == "weighted_sampler" and self.mode=="train"\
            and (not self.cfg['train']['two_stage']['drs'] or (self.cfg['train']['two_stage']['drs'] and self.epoch)):
            assert self.cfg['train']['sampler']['weighted_sampler']['type'] in ["balance", 'square', 'progressive']
            if self.cfg['train']['sampler']['weighted_sampler']['type'] == "balance":
                sample_class = random.randint(0, self.num_class - 1)
            elif self.cfg['train']['sampler']['weighted_sampler']['type'] == "square":
                sample_class = np.random.choice(np.arange(self.num_class), p=self.square_p)
            else:
                sample_class = np.random.choice(np.arange(self.num_class), p=self.progress_p)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        data = []
        label = self.data[self.label_name][index]
        meta = dict()
        for entity in self.entities:
            data.append(self.data[entity][index])
        label_weight = self.data[self.label_weight_name][index]
        lt_label = self.data[self.lt_label_name][index]

        meta_data, meta_label = None, None
        if (self.cfg['train']['sampler']['type'] == "bbn_sampler" and
            self.cfg['train']['sampler']['bbn_sampler']['type'] == "reverse") or \
                (self.cfg['train']['combiner']['type'] == "bbn_mix") and (self.mode == "train"):
            sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)
            sample_label = self.data[self.label_name][sample_index]
            sample_data = []
            for entity in self.entities:
                sample_data.append(self.data[entity][sample_index])
            # meta['sample_data'] = DeviceList(sample_data)
            # meta['sample_label'] = sample_label
            meta_data = DeviceList(sample_data)
            meta_label = sample_label

        item_dict = {'x': data, 'y': label, 'y_weight': label_weight, 'lt_class': lt_label, 'meta': meta,
                       'meta_data': meta_data, 'meta_label': meta_label}

        if self.cfg['loss']['type'] == 'LDS':
            item_dict['lds_weight'] = self.lds_weights[index]

        return item_dict

    def _get_class_dict(self):
        class_dict = dict()
        label_npy = np.array(self.data[self.label_name])
        for i in range(self.num_class):
            class_dict[i] = list(np.where(label_npy == i)[0].astype('int'))
        return class_dict

    def get_weight(self):
        num_list = utils.utils.get_category_list(self)
        max_num = max(num_list)
        class_weight = [max_num / i if i != 0 else 0 for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_class):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def update(self, epoch):
        self.epoch = max(0, epoch-self.cfg['train']['two_stage']['start_epoch']) if self.cfg['train']['two_stage']['drs'] else epoch
        if self.cfg['train']['sampler']['weighted_sampler']['type'] == "progressive":
            self.progress_p = epoch/self.cfg['train']['max_epoch'] * self.class_p + (1-epoch/self.cfg['train']['max_epoch'])*self.instance_p
            print('self.progress_p', self.progress_p)

    def get_lds_weights(self, reweight, max_target=121, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.data[self.label_name]

        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1

        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight

        print(f"Using re-weighting: [{reweight.upper()}]")

        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights



class DataLoader:

    """base data loader class that contains functions shared by almost all data loader classes.
    
    """
    
    def __init__(self):
        """empty data loader class, to be overwritten
        """
        pass

    def get_data(self, format='df'):
        '''
        Arguments:
            format (str, optional): the dataset format
                
        Returns:
            pd.DataFrame/dict/np.array: when format is df/dict/DeepPurpose
        
        Raises:
            AttributeError: format not supported
        '''
        if format == 'df':
            return pd.DataFrame({self.entity1_name + '_ID': self.entity1_idx,
                                 self.entity1_name: self.entity1, 'Y': self.y})
        elif format == 'dict':
            return {self.entity1_name + '_ID': self.entity1_idx,
                    self.entity1_name: self.entity1, 'Y': self.y}
        elif format == 'DeepPurpose':
            return self.entity1, self.y
        else:
            raise AttributeError("Please use the correct format input")

    def print_stats(self):
        """print statistics
        """
        print('There are ' + str(len(np.unique(
            self.entity1))) + ' unique ' + self.entity1_name.lower() + 's',
              flush=True, file=sys.stderr)

    def get_split(self, method='random', seed=42,
                  frac=[0.7, 0.1, 0.2]):
        '''
        split function, overwritten by single_pred/multi_pred/generation for more specific splits
        Arguments:
            method: splitting schemes
            seed: random seed
            frac: train/val/test split fractions
        
        Returns:
            dict: a dictionary of train/valid/test dataframes
        
        Raises:
            AttributeError: split method not supported 
        '''

        df = self.get_data(format='df')

        if method == 'random':
            return utils.create_fold(df, seed, frac)
        elif method == 'cold_' + self.entity1_name.lower():
            return utils.create_fold_setting_cold(df, seed, frac,
                                                  self.entity1_name)
        elif method == 'standard':
            # TODO: label_type default classification
            return utils.create_standard_fold(df, seed, frac, 'Y', np.unique(self.y).shape[0], 'classification', None)
        else:
            raise AttributeError("Please specify the correct splitting method")

    def label_distribution(self):
        """visualize distribution of labels
        """
        utils.label_dist(self.y, self.name)

    def binarize(self, threshold=None, order='descending'):
        """binarize the labels
        
        Args:
            threshold (float, optional): the threshold to binarize the label. 
            order (str, optional): the order of binarization, if ascending, flip 1 to larger values and vice versus for descending
        
        Returns:
            DataLoader: data loader class with updated label
        
        Raises:
            AttributeError: no threshold specified for binarization
        """
        if threshold is None:
            raise AttributeError(
                "Please specify the threshold to binarize the data by "
                "'binarize(threshold = N)'!")

        if (len(np.unique(self.y)) == 2):
            print("The data is already binarized!", flush=True,
                  file=sys.stderr)
        else:
            print("Binariztion using threshold " + str(
                threshold) + ", default, we assume the smaller values are 1 "
                             "and larger ones is 0, you can change the order "
                             "by 'binarize(order = 'ascending')'",
                  flush=True, file=sys.stderr)
            if np.unique(self.y).reshape(-1, ).shape[0] < 2:
                raise AttributeError(
                    "Adjust your threshold, there is only one class.")
            self.y = utils.binarize(self.y, threshold, order)
        return self

    def __len__(self):
        """get number of data points
        
        Returns:
            int: number of data points
        """
        return len(self.get_data(format='df'))

    def convert_to_log(self, form = 'standard'):
        """convert labels to log-scale
        
        Args:
            form (str, optional): standard log-transformation or binding nM <-> p transformation.
        """
        print('To log space...', flush=True, file=sys.stderr)
        if form == 'binding':
            self.y = utils.convert_to_log(self.y)
        elif form == 'standard':
            self.sign = np.sign(self.y)
            self.y = self.sign * np.log(abs(self.y) + 1e-10)

    def convert_from_log(self, form = 'standard'):
        """convert labels from log-scale
        
        Args:
            form (str, optional): standard log-transformation or binding nM <-> p transformation.
        """
        print('Convert Back To Original space...', flush=True, file=sys.stderr)
        if form == 'binding':
            self.y = utils.convert_back_log(self.y)
        elif form == 'standard':
            self.y = self.sign * (np.exp(self.sign * self.y) - 1e-10)

    def get_label_meaning(self, output_format='dict'):
        """get the biomedical meaning of label
        
        Args:
            output_format (str, optional): dict/df/array for label
        
        Returns:
            dict/pd.DataFrame/np.array: when output_format is dict/df/array
        """
        return utils.get_label_map(self.name, self.path, self.target,
                                   file_format=self.file_format,
                                   output_format=output_format)

    def zjhbalanced(self, oversample=False, seed=42):
        """balance the label neg-pos ratio
        
        Args:
            oversample (bool, optional): whether or not to oversample minority or subsample majority to match ratio
            seed (int, optional): random seed
        
        Returns:
            pd.DataFrame: the updated dataframe with balanced dataset
        
        Raises:
            AttributeError: alert to binarize the data first as continuous values cannot do balancing
        """
        if len(np.unique(self.y)) > 2:
            raise AttributeError(
                "You should binarize the data first by calling "
                "data.binarize(threshold)",
                flush=True, file=sys.stderr)

        val = self.get_data()

        class_ = val.Y.value_counts().keys().values
        major_class = class_[0]
        minor_class = class_[1]

        if not oversample:
            print(
                " Subsample the majority class is used, if you want to do "
                "oversample the minority class, set 'balanced(oversample = True)'. ",
                flush=True, file=sys.stderr)
            val = pd.concat(
                [val[val.Y == major_class].sample(
                    n=len(val[val.Y == minor_class]), replace=False,
                    random_state=seed), val[val.Y == minor_class]]).sample(
                frac=1,
                replace=False,
                random_state=seed).reset_index(
                drop=True)
        else:
            print(" Oversample of minority class is used. ", flush=True,
                  file=sys.stderr)
            val = pd.concat(
                [val[val.Y == minor_class].sample(
                    n=len(val[val.Y == major_class]), replace=True,
                    random_state=seed), val[val.Y == major_class]]).sample(
                frac=1,
                replace=False,
                random_state=seed).reset_index(
                drop=True)
        return val