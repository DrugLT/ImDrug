# -*- coding: utf-8 -*-
# Author: ImDrug Team
# License: MIT

import warnings
warnings.filterwarnings("ignore")
import sys

from utils import print_sys
from . import bi_pred_dataset, multi_pred_dataset
from metadata import dataset_names

class PPI(bi_pred_dataset.DataLoader):

    """Data loader class to load datasets in Protein-Protein Interaction Prediction task. 
    More info: https://tdcommons.ai/multi_pred_tasks/ppi/

    Task Description: Binary Classification. Given the target amino acid sequence pairs, predict if they interact or not.


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
        """Create Protein-Protein Interaction Prediction dataloader object
        """
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["PPI"])
        self.entity1_name = 'Protein1'
        self.entity2_name = 'Protein2'
        self.two_types = False
        self.entities = [self.entity1_name, self.entity2_name]
        self.entity_type = ['protein', 'protein']

        if print_stats:
            self.print_stats()

        print('Done!', flush=True, file=sys.stderr)

    def print_stats(self):
        """print the statistics of the dataset
        """
        import numpy as np
        print_sys('--- Dataset Statistics ---')
        print('There are ' + str(len(np.unique(
            self.entity1.tolist() + self.entity2.tolist()))) + ' unique proteins.',
              flush=True, file=sys.stderr)
        print('There are ' + str(len(self.y)) + ' protein-protein pairs.',
              flush=True, file=sys.stderr)
        print_sys('--------------------------')