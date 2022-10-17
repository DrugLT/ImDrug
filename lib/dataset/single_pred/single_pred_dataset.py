import pandas as pd
import numpy as np
import os, sys, json
import warnings
warnings.filterwarnings("ignore")

from .. import base_dataset

from utils import dataset2target_lists, \
					property_dataset_load,\
					create_open_split_byclass, \
					create_fold, \
					create_fold_byclass, \
					create_fold_time, \
					create_fold_time_byclass, \
					create_fold_setting_cold, \
					create_fold_setting_cold_byclass, \
					create_combination_split, \
					create_combination_split_by_class, \
					create_scaffold_split, \
					create_scaffold_split_byclass, \
					create_standard_fold, \
					print_sys

class DataLoader(base_dataset.DataLoader):

	"""A base data loader class.

	Args:
		name (str): The dataset name.
		path (str): The path to save the data file
		label_name (str): For multi-label dataset, specify the label name
		print_stats (bool): Whether to print basic statistics of the dataset
		dataset_names (list): A list of dataset names available for a task
		convert_format (str): Automatic conversion of SMILES to other molecular formats in MolConvert class. Stored as separate column in dataframe

	Attributes:
		convert_format (str): conversion format of an entity
		convert_result (list): a placeholder for a list of conversion outputs
		entity1 (Pandas Series): a list of the single entities
		entity1_idx (Pandas Series): a list of the single entity indices 
		entity_name (Pandas Series): a list of the single entity names
	    file_format (str): the format of the downloaded dataset
	    label_name (str): for multi-label dataset, the label name of interest
	    name (str): dataset name
	    path (str): path to save and retrieve the dataset
	    y (Pandas Series): a list of the single entities label 
	"""

	def __init__(self, name, path, label_name, print_stats, dataset_names, convert_format, raw_format = 'SMILES'):
		"""Create a base dataloader object that each single instance prediction task dataloader class can inherit from.

		Raises:		
			ValueError: for a dataset with multiple labels, specify the label. Use to druglt.utils.retrieve_label_name see the available label names
		
		"""
		# if name.lower() in dataset2target_lists.keys():
		# 	if label_name is None:
		# 		raise ValueError("Please select a label name. You can use druglt.utils.retrieve_label_name_list('" + name.lower() + "') to retrieve all available label names.")


		entity1, y, entity1_idx = property_dataset_load(name, path, label_name, dataset_names)

		self.entity1 = entity1
		self.y = y
		self.entity1_idx = entity1_idx
		self.name = name
		self.entity1_name = 'Drug'
		self.entities = [self.entity1_name]
		self.entity_type = ['drug']
		self.path = path
		self.file_format = 'csv'
		self.label_name = label_name
		self.convert_format = convert_format
		self.convert_result = None
		self.raw_format = raw_format ### 'SMILES' for most data, 'Raw3D' for QM9, ...

	def get_data(self, format='df', label_name='Y'):
		'''
		Arguments:
			format (str, optional): the returning dataset format, defaults to 'df'

		Returns:
			pandas DataFrame/dict: a dataframe of a dataset/a dictionary for key information in the dataset

		Raises:
			AttributeError: use the correct format input (df, dict, DeepPurpose)
		'''

		if (self.convert_format is not None) and (self.convert_result is None):
			from chem_utils import MolConvert
			converter = MolConvert(src = self.raw_format, dst = self.convert_format)
			convert_result = converter(self.entity1.values)
			self.convert_result = [i for i in convert_result]
		
		if format == 'df':
			if self.convert_format is not None:
				return pd.DataFrame({self.entity1_name + '_ID': self.entity1_idx, self.entity1_name: self.entity1, self.entity1_name + '_' + self.convert_format: self.convert_result, label_name: self.y})
			else:
				return pd.DataFrame({self.entity1_name + '_ID': self.entity1_idx, self.entity1_name: self.entity1, label_name: self.y})
		elif format == 'dict':
			if self.convert_format is not None:
				return {self.entity1_name + '_ID': self.entity1_idx.values, self.entity1_name: self.entity1.values, self.entity1_name + '_' + self.convert_format: self.convert_result, label_name: self.y.values}
			else:
				return {self.entity1_name + '_ID': self.entity1_idx.values, self.entity1_name: self.entity1.values, label_name: self.y.values}
		elif format == 'DeepPurpose':
			return self.entity1.values, self.y.values
		else:
			raise AttributeError("Please use the correct format input")
	
	def get_split(self, method = 'random', seed = 42, frac = [0.7, 0.1, 0.2], lt_frac=[0.4, 0.4, 0.2], open_frac=0.2, 
	              column_name=None, time_column=None, label_name='Y', label_weight_name='Y_Weight', lt_label_name='LT_Class', 
				num_class=10, label_type='classification', scale=None, by_class=True):
		'''
		Arguments:
			method: splitting schemes, choose from random, cold_{entity}, scaffold, defaults to 'random'
			seed: the random seed for splitting dataset, defaults to '42'
			frac: train/val/test split fractions, defaults to '[0.7, 0.1, 0.2]'
		
		Returns:
			dict: a dictionary with three keys ('train', 'valid', 'test'), each value is a pandas dataframe object of the splitted dataset
		
		Raises:
		 	AttributeError：the input split method is not available.
		'''

		df = self.get_data(format='df', label_name=label_name)


		if by_class: # split class by class
			if method.startswith('open-'):
				closed_df, open_df = create_open_split_byclass(df=df, open_frac=open_frac, label_name=label_name, label_weight_name=label_weight_name, lt_label_name=lt_label_name,
																num_class=num_class, label_type=label_type, scale=scale)
				method = method.replace('open-', '')
				if method == 'random':
					data_dic = create_fold_byclass(df=closed_df, seed=seed, frac=frac, lt_frac=lt_frac, label_name=label_name,  label_weight_name=label_weight_name, lt_label_name=lt_label_name, 
												num_class=num_class, label_type=label_type, scale=scale)

				elif method == 'cold_' + self.entity1_name.lower():
					data_dic = create_fold_setting_cold_byclass(df=closed_df, seed=seed, frac=frac, lt_frac=lt_frac, entities=self.entity1_name, label_name=label_name, 
															label_weight_name=label_weight_name, lt_label_name=lt_label_name, num_class=num_class, label_type=label_type, scale=scale)
				elif method == 'scaffold':
					data_dic = create_scaffold_split_byclass(df=closed_df, seed=seed, frac=frac, lt_frac=lt_frac, entity=self.entity1_name, label_name=label_name,  label_weight_name=label_weight_name, 
															lt_label_name=lt_label_name, num_class=num_class, label_type=label_type, scale=scale)
				elif method == 'time':
					data_dic = create_fold_time_byclass(df=closed_df, frac=frac, time_column=time_column, lt_frac=lt_frac, label_name=label_name,  label_weight_name=label_weight_name, lt_label_name=lt_label_name,
													num_class=num_class, label_type=label_type, scale=scale)
				elif method == 'standard':
					data_dic = create_standard_fold(df=closed_df, fold_seed=seed, frac=frac, lt_frac=lt_frac, label_name=label_name, label_weight_name=label_weight_name, lt_label_name=lt_label_name, num_class=num_class, label_type=label_type, scale=scale)
				else:
					raise AttributeError("Please select a splitting strategy from random, cold_split, or combination.")

				data_dic['test'] = data_dic['test'].append(open_df) # merge test set for closed and open classes
				return data_dic
			else:
				if method == 'random':
					return create_fold_byclass(df=df, seed=seed, frac=frac, lt_frac=lt_frac, label_name=label_name,  label_weight_name=label_weight_name, lt_label_name=lt_label_name, 
												num_class=num_class, label_type=label_type, scale=scale)

				elif method == 'cold_' + self.entity1_name.lower():
					return create_fold_setting_cold_byclass(df=df, seed=seed, frac=frac, lt_frac=lt_frac, entities=self.entity1_name, label_name=label_name, 
															label_weight_name=label_weight_name, lt_label_name=lt_label_name, num_class=num_class, label_type=label_type, scale=scale)
				elif method == 'scaffold':
					return create_scaffold_split_byclass(df=df, seed=seed, frac=frac, lt_frac=lt_frac, entity=self.entity1_name, label_name=label_name,  label_weight_name=label_weight_name, 
															lt_label_name=lt_label_name, num_class=num_class, label_type=label_type, scale=scale)
				elif method == 'time':
					return create_fold_time_byclass(df=df, frac=frac, time_column=time_column, lt_frac=lt_frac, label_name=label_name,  label_weight_name=label_weight_name, lt_label_name=lt_label_name,
													num_class=num_class, label_type=label_type, scale=scale)
				elif method == 'standard':
					return create_standard_fold(df=df, fold_seed=seed, frac=frac, lt_frac=lt_frac, label_name=label_name, label_weight_name=label_weight_name, lt_label_name=lt_label_name, num_class=num_class, label_type=label_type, scale=scale)
				else:
					raise AttributeError("Please select a splitting strategy from random, cold_split, or combination.")

		else:
			if method == 'random':
				return create_fold(df, seed, frac)
			elif method == 'cold_' + self.entity1_name.lower():
				return create_fold_setting_cold(df, seed, frac, self.entity1_name)
			elif method == 'scaffold':
				return create_scaffold_split(df, seed, frac, self.entity1_name)
			elif method == 'time':
				return create_fold_time(df, frac, time_column)
			elif method == 'standard':
				return create_standard_fold(df=df, fold_seed=seed, frac=frac, lt_frac=lt_frac, label_name=label_name, label_weight_name=label_weight_name, lt_label_name=lt_label_name, num_class=num_class, label_type=label_type, scale=scale)
			else:
				raise AttributeError("Please specify the correct splitting method")
	
	def print_stats(self):
		"""Print basic data statistics.
		"""
		print_sys('--- Dataset Statistics ---')
		try:
			x = np.unique(self.entity1)
		except:
			x = np.unique(self.entity1_idx)
		
		print(str(len(x)) + 'unique' + self.entity1_name.lower() + 's.', flux = True, file = sys.stderr)
		print_sys('------------------------')