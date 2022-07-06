# -*- coding: utf-8 -*-
# Author: ImDrug Team
# License: MIT

import warnings
warnings.filterwarnings("ignore")
import sys

from utils import print_sys
from . import bi_pred_dataset, multi_pred_dataset
from metadata import dataset_names

class ReactType(bi_pred_dataset.DataLoader):

    """Data loader class to load datasets in Reaction Type Prediction task   

    Task Description: Given reactant and product set X, predict the reation type Y.


    Args:
        name (str): the dataset name.
        path (str, optional): 
            The path to save the data file, defaults to './data'
        label_name (str, optional): 
            For multi-label dataset, specify the label name, defaults to None
        print_stats (bool, optional): 
            Whether to print basic statistics of the dataset, defaults to False

    """
    
    def __init__(self, name, path='./data', label_name=None,
                 print_stats=False):
        """Create Reaction Type Prediction dataloader object
        """
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["ReactType"])
        self.entity1_name = 'Reactant'
        self.entity2_name = 'Product'
        self.entities = [self.entity1_name, self.entity2_name]
        self.entity_type = ['drug', 'drug']
        self.two_types = True
        
        if print_stats:
            self.print_stats()

        print('Done!', flush=True, file=sys.stderr)
