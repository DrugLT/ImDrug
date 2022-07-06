# -*- coding: utf-8 -*-
# Author: ImDrug Team
# License: MIT

"""Utilities functions for splitting dataset 
"""
import os, sys
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

def get_cls_labels(df, label_name, num_class, label_type, scale):
	if label_type == "classification":
		hist = df[label_name].value_counts() # automatically sorted by value count
		cls_labels = hist.index.to_numpy()
		print('Value counts for LT Classification: \n', hist)
		print('Class labels for LT Classification', cls_labels)
		print('Imbalance Ratio for LT Classification', hist.to_list()[0]/hist.to_list()[-1])
		return cls_labels, np.array(list(cls_labels)), hist.to_numpy()
	elif label_type == "regression":
		reg_labels = df[label_name].to_list()
		if scale == 'log':
			reg_labels = np.log(reg_labels)
		hist, bins =  np.histogram(reg_labels, bins=num_class) # bins are monotonically increasing by default
		print('Hist for LT Regression', hist)
		print('Bins for LT Regression', bins)
		bins[0] -= 1e-8 # include boundary samples
		bins[-1] += 1e8 # include boundary samples
		reg_cls_labels = np.digitize(reg_labels, bins=bins) - 1 # length: n_sample
		df['reg_cls'] = list(reg_cls_labels)

		hist = pd.Series(reg_cls_labels).value_counts() # automatically sorted by value count
		cls_labels = hist.index.to_numpy() # length: n_class
		print('Value counts for LT Regression: \n', hist)
		print('Class labels for LT Regression', cls_labels)
		print('Imbalance Ratio for LT Regression', hist.to_list()[0]/hist.to_list()[-1])
		return cls_labels, np.array(list(cls_labels)), hist.to_numpy()

	else:
		raise NotImplementedError

def get_lt_labels(cls_labels, lt_frac):
	if len(cls_labels) == 0:
		return []
	else:
		lt_labels = []
		num_head = int(np.ceil(len(cls_labels) * lt_frac[0]))
		num_middle_tail = len(cls_labels) - num_head 
		for i in range(num_head):
			lt_labels.append('head{}'.format(i))
		if num_middle_tail == 0:
			return lt_labels
		else:
			# num_tail = int(np.ceil(num_middle_tail * lt_frac[-1]/(lt_frac[1] + lt_frac[-1]))) # note there might be over estimation due to rounding errors: e.g., num_class=10, lt_frac=[0.3, 0.4, 0.3], num_middle_tail * lt_frac[-1]/(lt_frac[1] + lt_frac[-1]) = 3.0000000000000004
			num_tail = int(np.rint(num_middle_tail * lt_frac[-1]/(lt_frac[1] + lt_frac[-1])))
			num_middle = num_middle_tail - num_tail 
			if num_middle > 0:
				for i in range(num_middle):
					lt_labels.append('middle{}'.format(i))
			for i in range(num_tail):
				lt_labels.append('tail{}'.format(i))
		return lt_labels


def get_label_weight(df, label_weight_name='Y_Weight'):
	if len(df) > 0:
		df[label_weight_name] = 1/len(df)
	return df

def create_open_split_byclass(df, open_frac, label_name='Y', label_weight_name='Y_Weight', lt_label_name='LT_Class', num_class=10, label_type='classification', scale=None):
	"""split whole dataset into closed set and open set (for testing only), class by class

	Args:
		frac (float): fraction of number of open classes

	Returns:
	"""
	cls_labels, bins, hist = get_cls_labels(df=df, label_name=label_name, num_class=num_class, label_type=label_type, scale=scale)
	num_open_classes = int(np.ceil(len(cls_labels) * open_frac))
	# open_cls_indices = np.argpartition(hist, num_open_classes)[:num_open_classes]
	open_cls_indices = cls_labels[-num_open_classes:]
	print(hist, open_cls_indices)

	df_cpy = copy.deepcopy(df)
	for i, open_cls_idx in enumerate(open_cls_indices):
		df_cpy[lt_label_name] = 'open{}'.format(i)

	if label_type == "regression":
		label_name = "reg_cls"
	open_df = df_cpy[df_cpy[label_name].isin(open_cls_indices)]

	# regard all open classes as one unknown class, as in https://arxiv.org/pdf/1904.05160.pdf
	open_df = get_label_weight(open_df, label_weight_name)

	closed_df = df[~df.index.isin(open_df.index)]
	return closed_df, open_df


def create_fold_byclass(df, seed, frac, lt_frac, label_name='Y', label_weight_name='Y_Weight', lt_label_name='LT_Class', num_class=10, label_type='classification', scale=None):
	"""create random split class by class
	
	Args:
	    df (pd.DataFrame): dataset dataframe
	    fold_seed (int): the random seed
	    frac (list): a list of train/valid/test fractions
	
	Returns:
	    dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""

	cls_labels, bins, hist = get_cls_labels(df=df, label_name=label_name, num_class=num_class, label_type=label_type, scale=scale)
	lt_labels = get_lt_labels(cls_labels, lt_frac)
	train_frac, val_frac, test_frac = frac
	train = pd.DataFrame()
	val = pd.DataFrame()
	test = pd.DataFrame()

	if label_type == "regression":
		label_name = "reg_cls"
	for cls_label, lt_label in zip(cls_labels, lt_labels):
		sub_df = df[df[label_name] == cls_label].copy()
		
		sub_test = sub_df.sample(frac = test_frac, replace = False, random_state = seed)
		train_val = sub_df[~sub_df.index.isin(sub_test.index)]
		sub_val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = seed)
		sub_train = train_val[~train_val.index.isin(sub_val.index)]

		sub_train = get_label_weight(sub_train, label_weight_name)
		sub_val = get_label_weight(sub_val, label_weight_name)
		sub_test = get_label_weight(sub_test, label_weight_name)

		sub_train[lt_label_name] = lt_label
		sub_val[lt_label_name] = lt_label
		sub_test[lt_label_name] = lt_label

		train = train.append(sub_train)
		val = val.append(sub_val)
		test = test.append(sub_test)

	return {'train': train.reset_index(drop = True),
			'valid': val.reset_index(drop = True),
			'test': test.reset_index(drop = True),
			'bins': bins,
			'hist': hist}

def create_fold_setting_cold_byclass(df, fold_seed, frac, lt_frac, entities, label_name='Y', label_weight_name='Y_Weight', lt_label_name='LT_Class', num_class=10, label_type='classification', scale=None):
	"""create cold-split where given one or multiple columns, it first splits based on
	entities in the columns and then maps all associated data points to the partition, class by class
	Args:
		df (pd.DataFrame): dataset dataframe
		fold_seed (int): the random seed
		frac (list): a list of train/valid/test fractions
		entities (Union[str, List[str]]): either a single "cold" entity or a list of
			"cold" entities on which the split is done
	Returns:
		dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""
	cls_labels, bins, hist = get_cls_labels(df=df, label_name=label_name, num_class=num_class, label_type=label_type, scale=scale)
	lt_labels = get_lt_labels(cls_labels, lt_frac)
	if isinstance(entities, str):
		entities = [entities]

	train_frac, val_frac, test_frac = frac

	train = pd.DataFrame()
	val = pd.DataFrame()
	test = pd.DataFrame()

	if label_type == "regression":
		label_name = "reg_cls"
	for cls_label, lt_label in zip(cls_labels, lt_labels):
		sub_df = df[df[label_name] == cls_label].copy()

		# For each entity, sample the instances belonging to the test datasets
		test_entity_instances = [
			sub_df[e].drop_duplicates().sample(
				frac=test_frac, replace=False, random_state=fold_seed
			).values for e in entities
		]

		# Select samples where all entities are in the test set
		sub_test = sub_df.copy()
		for entity, instances in zip(entities, test_entity_instances):
			sub_test = sub_test[sub_test[entity].isin(instances)]

		if len(sub_df) > 0 and len(sub_test) == 0:
			print(sub_df)
			raise ValueError(
				'No test samples found. Try another seed, increasing the test frac or a '
				'less stringent splitting strategy.'
			)

		# Proceed with validation data
		train_val = sub_df.copy()
		for i, e in enumerate(entities):
			train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

		val_entity_instances = [
			train_val[e].drop_duplicates().sample(
				frac=val_frac/(1-test_frac), replace=False, random_state=fold_seed
			).values for e in entities
		]
		sub_val = train_val.copy()
		for entity, instances in zip(entities, val_entity_instances):
			sub_val = sub_val[sub_val[entity].isin(instances)]

		if len(sub_df) > 0 and len(sub_val) == 0:
			raise ValueError(
				'No validation samples found. Try another seed, increasing the test frac '
				'or a less stringent splitting strategy.'
			)

		sub_train = train_val.copy()
		for i,e in enumerate(entities):
			sub_train = sub_train[~sub_train[e].isin(val_entity_instances[i])]

		sub_train = get_label_weight(sub_train, label_weight_name)
		sub_val = get_label_weight(sub_val, label_weight_name)
		sub_test = get_label_weight(sub_test, label_weight_name)

		sub_train[lt_label_name] = lt_label
		sub_val[lt_label_name] = lt_label
		sub_test[lt_label_name] = lt_label

		train = train.append(sub_train)
		val = val.append(sub_val)
		test = test.append(sub_test)

	return {'train': train.reset_index(drop = True),
			'valid': val.reset_index(drop = True),
			'test': test.reset_index(drop = True),
			'bins': bins,
			'hist': hist}


def create_scaffold_split_byclass(df, seed, frac, lt_frac, entity, label_name='Y',label_weight_name='Y_Weight', lt_label_name='LT_Class', num_class=10, label_type='classification', scale=None):
	"""create scaffold split. it first generates molecular scaffold for each molecule and then split based on scaffolds, class by class
	reference: https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py
	Args:
	    df (pd.DataFrame): dataset dataframe
	    fold_seed (int): the random seed
	    frac (list): a list of train/valid/test fractions
	    entity (str): the column name for where molecule stores
	
	Returns:
	    dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""
	try:
		from rdkit import Chem
		from rdkit.Chem.Scaffolds import MurckoScaffold
		from rdkit import RDLogger
		RDLogger.DisableLog('rdApp.*')
	except:
		raise ImportError("Please install rdkit by 'conda install -c conda-forge rdkit'! ")
	from tqdm import tqdm
	from random import Random

	from collections import defaultdict
	random = Random(seed)
	df = copy.deepcopy(df)
	cls_labels, bins, hist = get_cls_labels(df=df, label_name=label_name, num_class=num_class, label_type=label_type, scale=scale)
	lt_labels = get_lt_labels(cls_labels, lt_frac)
	train, val, test = [], [], []
	df[label_weight_name] = 0

	if label_type == "regression":
		label_name = "reg_cls"
	for cls_label, lt_label in zip(cls_labels, lt_labels):
		sub_df = df[df[label_name] == cls_label].copy()
	
		s = sub_df[entity].values
		scaffolds = defaultdict(set)
		idx2mol = dict(zip(list(range(len(s))),s))

		error_smiles = 0
		for i, smiles in tqdm(enumerate(s), total=len(s)):
			try:
				scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol = Chem.MolFromSmiles(smiles), includeChirality = False)
				scaffolds[scaffold].add(i)
			except:
				print_sys(smiles + ' returns RDKit error and is thus omitted...')
				error_smiles += 1

		sub_train, sub_val, sub_test = [], [], []
		train_size = int((len(sub_df) - error_smiles) * frac[0])
		val_size = int((len(sub_df) - error_smiles) * frac[1])
		test_size = (len(sub_df) - error_smiles) - train_size - val_size
		train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

		#index_sets = sorted(list(scaffolds.values()), key=lambda i: len(i), reverse=True)
		index_sets = list(scaffolds.values())
		big_index_sets = []
		small_index_sets = []
		for index_set in index_sets:
			if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
				big_index_sets.append(index_set)
			else:
				small_index_sets.append(index_set)
		random.seed(seed)
		random.shuffle(big_index_sets)
		random.shuffle(small_index_sets)
		index_sets = big_index_sets + small_index_sets

		if frac[2] == 0:
			for index_set in index_sets:
				if len(sub_train) + len(index_set) <= train_size:
					sub_train += index_set
					train_scaffold_count += 1
				else:
					sub_val += index_set
					val_scaffold_count += 1
		else:
			for index_set in index_sets:
				if len(sub_train) + len(index_set) <= train_size:
					sub_train += index_set
					train_scaffold_count += 1
				elif len(sub_val) + len(index_set) <= val_size:
					sub_val += index_set
					val_scaffold_count += 1
				else:
					sub_test += index_set
					test_scaffold_count += 1

		df.iloc[sub_train][label_weight_name] = 0 if len(sub_train) == 0 else 1/len(sub_train)
		df.iloc[sub_val][label_weight_name] = 0 if len(sub_val) == 0 else 1/len(sub_val)
		df.iloc[sub_test][label_weight_name] = 0 if len(sub_test) == 0 else 1/len(sub_test)

		df.iloc[sub_train][lt_label_name] = lt_label
		df.iloc[sub_val][lt_label_name] = lt_label
		df.iloc[sub_test][lt_label_name] = lt_label

		train += sub_train
		val += sub_val
		test += sub_test

	return {'train': df.iloc[train].reset_index(drop = True),
			'valid': df.iloc[val].reset_index(drop = True),
			'test': df.iloc[test].reset_index(drop = True),
			'bins': bins,
			'hist': hist}

def create_combination_split_by_class(df, seed, frac, lt_frac, label_name='Y', label_weight_name='Y_Weight', lt_label_name='LT_Class', num_class=10, label_type='classification', scale=None):
	"""
	Function for splitting drug combination dataset such that no combinations are shared across the split, class by class
	
	Args:
	    df (pd.Dataframe): dataset to split
	    seed (int): random seed
	    frac (list): split fraction as a list
	
	Returns:
	    dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""
	
	cls_labels, bins, hist = get_cls_labels(df=df, label_name=label_name, num_class=num_class, label_type=label_type, scale=scale)
	lt_labels = get_lt_labels(cls_labels, lt_frac)

	train_set, val_set, test_set = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

	if label_type == "regression":
		label_name = "reg_cls"
	for cls_label, lt_label in zip(cls_labels, lt_labels):
		sub_df = df[df[label_name] == cls_label].copy()
		test_size = int(len(sub_df) * frac[2])
		train_size = int(len(sub_df) * frac[0])
		val_size = len(sub_df) - train_size - test_size
		np.random.seed(seed)

		# Create a new column for combination names
		sub_df['concat'] = sub_df['Drug1_ID'] + ',' + sub_df['Drug2_ID']

		# Identify shared drug combinations across all target classes
		combinations = []
		for c in sub_df['Cell_Line_ID'].unique():
			df_cell = sub_df[sub_df['Cell_Line_ID'] == c]
			combinations.append(set(df_cell['concat'].values))

		intxn = combinations[0]
		for c in combinations:
			intxn = intxn.intersection(c)

		# Split combinations into train, val and test
		test_choices = np.random.choice(list(intxn),
							min(len(intxn), int(test_size / len(sub_df['Cell_Line_ID'].unique()))),
							replace=False)
		trainval_intxn = intxn.difference(test_choices)
		val_choices = np.random.choice(list(trainval_intxn),
							min(len(trainval_intxn), int(val_size / len(sub_df['Cell_Line_ID'].unique()))),
							replace=False)

		## Create train and test set
		test_subset = sub_df[sub_df['concat'].isin(test_choices)].drop(columns=['concat'])
		val_subset = sub_df[sub_df['concat'].isin(val_choices)]
		train_subset = sub_df[~sub_df['concat'].isin(test_choices)].reset_index(drop=True)
		train_subset = train_subset[~train_subset['concat'].isin(val_choices)]

		train_subset = get_label_weight(train_subset, label_weight_name)
		val_subset = get_label_weight(val_subset, label_weight_name)
		test_subset = get_label_weight(test_subset, label_weight_name)

		train_subset[lt_label_name] = lt_label
		val_subset[lt_label_name] = lt_label
		test_subset[lt_label_name] = lt_label

		train_set = train_set.append(train_subset)
		val_set = val_set.append(val_subset)
		test_set = test_set.append(test_subset)

	return {'train': train_set.reset_index(drop = True),
			'valid': val_set.reset_index(drop = True),
			'test': test_set.reset_index(drop = True),
			'bins': bins,
			'hist': hist}

# create time split

def create_fold_time_byclass(df, frac, date_column, lt_frac, label_name='Y', label_weight_name='Y_Weight', lt_label_name='LT_Class', num_class=10, label_type='classification', scale=None):
	"""create splits based on time, class by class
	
	Args:
	    df (pd.DataFrame): the dataset dataframe
	    frac (list): list of train/valid/test fractions
	    date_column (str): the name of the column that contains the time info
	
	Returns:
	    dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""
	cls_labels, bins, hist = get_cls_labels(df=df, label_name=label_name, num_class=num_class, label_type=label_type, scale=scale)
	lt_labels = get_lt_labels(cls_labels, lt_frac)

	train, val, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

	split_time_dic = {'train_time_frame': {},
				      'valid_time_frame': {},
				  	  'test_time_frame': {}
					}

	if label_type == "regression":
		label_name = "reg_cls"
	for cls_label, lt_label in zip(cls_labels, lt_labels):
		sub_df = df[df[label_name] == cls_label].copy()
		
		if len(sub_df) > 0:
			sub_df = sub_df.sort_values(by = date_column).reset_index(drop = True)
			train_frac, val_frac, test_frac = frac[0], frac[1], frac[2]

			split_date = sub_df[:int(np.ceil(len(sub_df) * (train_frac + val_frac)))].iloc[-1][date_column]
			sub_test = sub_df[sub_df[date_column] >= split_date].reset_index(drop = True)
			train_val = sub_df[sub_df[date_column] < split_date]

			if len(train_val) == 0:
				print('# of training and validation for cls: {} is zero'.format(cls_label))
				split_date_valid = split_date
				sub_train = pd.DataFrame()
				sub_val = pd.DataFrame()
			elif int(len(train_val) * train_frac/(train_frac + val_frac)) == 0:
				print('# of validation for cls: {} is zero'.format(cls_label))
				split_date_valid = split_date
				sub_val = pd.DataFrame()
				sub_train = train_val[train_val[date_column] <= split_date_valid].reset_index(drop = True)
			else:
				split_date_valid = train_val[:int(len(train_val) * train_frac/(train_frac + val_frac))].iloc[-1][date_column]
				sub_train = train_val[train_val[date_column] <= split_date_valid].reset_index(drop = True)
				sub_val = train_val[train_val[date_column] > split_date_valid].reset_index(drop = True)

			sub_train = get_label_weight(sub_train, label_weight_name)
			sub_val = get_label_weight(sub_val, label_weight_name)
			sub_test = get_label_weight(sub_test, label_weight_name)

			sub_train[lt_label_name] = lt_label
			sub_val[lt_label_name] = lt_label
			sub_test[lt_label_name] = lt_label

			train = train.append(sub_train)
			val = val.append(sub_val)
			test = test.append(sub_test)

		split_time_dic['train_time_frame'][cls_label] = (sub_df.iloc[0][date_column], split_date_valid)
		split_time_dic['valid_time_frame'][cls_label] = (split_date_valid, split_date)
		split_time_dic['test_time_frame'][cls_label] = (split_date, sub_df.iloc[-1][date_column])

	return {'train': train, 'valid': val, 'test': test, 'split_time': split_time_dic, 'bins': bins, 'hist': hist}

def get_balanced_dataset(df, lt_frac, label_name='Y', label_weight_name='Y_Weight', lt_label_name='LT_Class', num_class=10, label_type='classification', scale=None):
	cls_labels, bins, hist = get_cls_labels(df=df, label_name=label_name, num_class=num_class, label_type=label_type, scale=scale)
	lt_labels = get_lt_labels(cls_labels, lt_frac)
	min_num = hist.min()
	df[label_weight_name] = 1/min_num
	idx_list = []
	if label_type == "regression":
		label_name = "reg_cls"
	df[lt_label_name] = ['none']*df.shape[0]
	for cls_label, lt_label in zip(cls_labels, lt_labels):
		l = list(df[df[label_name] == cls_label].index)
		df.loc[df[label_name] == cls_label, lt_label_name] = lt_label 
		if df[df[label_name] == cls_label].shape[0] > min_num:
			l_new = np.random.choice(l, min_num, replace=False).tolist()
			idx_list += l_new
		else:
			idx_list += l

	return df.loc[idx_list]


def create_standard_fold(df, fold_seed, frac, lt_frac, label_name='Y', label_weight_name='Y_Weight', lt_label_name='LT_Class', num_class=10, label_type='classification', scale=None):
	"""create standard split:
		imbalanced training dataset and balanced validation and testing dataset

	Args:
	    df (pd.DataFrame): dataset dataframe
	    fold_seed (int): the random seed
	    frac (list): a list of train/valid/test fractions
	
	Returns:
	    dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""
	train_frac, val_frac, test_frac = frac
	test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
	train_val = df[~df.index.isin(test.index)]
	val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed)
	train = train_val[~train_val.index.isin(val.index)]
	train = get_label_weight(train, label_weight_name)

	cls_labels, bins, hist = get_cls_labels(df=train, label_name=label_name, num_class=num_class, label_type=label_type, scale=scale)
	lt_labels = get_lt_labels(cls_labels, lt_frac)
	train[lt_label_name] = ['none']*train.shape[0]
	for cls_label, lt_label in zip(cls_labels, lt_labels):
		train.loc[train[label_name] == cls_label, lt_label_name] = lt_label


	val.index, test.index = np.arange(val.shape[0]), np.arange(test.shape[0])
	val_new = get_balanced_dataset(val, lt_frac, label_name=label_name, num_class=num_class, label_type=label_type, scale=scale)
	test_new = get_balanced_dataset(test, lt_frac, label_name=label_name, num_class=num_class, label_type=label_type, scale=scale)

	cls_labels, bins, hist = get_cls_labels(df=df, label_name=label_name, num_class=num_class, label_type=label_type, scale=scale)
	lt_labels = get_lt_labels(cls_labels, lt_frac)

	if label_type == "regression":
		label_name = "reg_cls"
	for cls_label, lt_label in zip(cls_labels, lt_labels):
		train[train[label_name] == cls_label][label_weight_name] = 1/len(train[train[label_name] == cls_label].index)
		train[train[label_name] == cls_label][lt_label_name] = lt_label


	return {'train': train.reset_index(drop = True),
			'valid': val_new.reset_index(drop = True),
			'test': test_new.reset_index(drop = True),
			'hist': hist}


def create_fold(df, fold_seed, frac):
	"""create random split
	
	Args:
	    df (pd.DataFrame): dataset dataframe
	    fold_seed (int): the random seed
	    frac (list): a list of train/valid/test fractions
	
	Returns:
	    dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""
	train_frac, val_frac, test_frac = frac
	test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
	train_val = df[~df.index.isin(test.index)]
	val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed)
	train = train_val[~train_val.index.isin(val.index)]

	return {'train': train.reset_index(drop = True),
			'valid': val.reset_index(drop = True),
			'test': test.reset_index(drop = True)}

def create_fold_setting_cold(df, fold_seed, frac, entities):
	"""create cold-split where given one or multiple columns, it first splits based on
	entities in the columns and then maps all associated data points to the partition

	Args:
		df (pd.DataFrame): dataset dataframe
		fold_seed (int): the random seed
		frac (list): a list of train/valid/test fractions
		entities (Union[str, List[str]]): either a single "cold" entity or a list of
			"cold" entities on which the split is done

	Returns:
		dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""
	if isinstance(entities, str):
		entities = [entities]

	train_frac, val_frac, test_frac = frac

	# For each entity, sample the instances belonging to the test datasets
	test_entity_instances = [
		df[e].drop_duplicates().sample(
			frac=test_frac, replace=False, random_state=fold_seed
		).values for e in entities
	]

	# Select samples where all entities are in the test set
	test = df.copy()
	for entity, instances in zip(entities, test_entity_instances):
		test = test[test[entity].isin(instances)]

	if len(test) == 0:
		raise ValueError(
			'No test samples found. Try another seed, increasing the test frac or a '
			'less stringent splitting strategy.'
		)

	# Proceed with validation data
	train_val = df.copy()
	for i, e in enumerate(entities):
		train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

	val_entity_instances = [
		train_val[e].drop_duplicates().sample(
			frac=val_frac/(1-test_frac), replace=False, random_state=fold_seed
		).values for e in entities
	]
	val = train_val.copy()
	for entity, instances in zip(entities, val_entity_instances):
		val = val[val[entity].isin(instances)]

	if len(val) == 0:
		raise ValueError(
			'No validation samples found. Try another seed, increasing the test frac '
			'or a less stringent splitting strategy.'
		)

	train = train_val.copy()
	for i,e in enumerate(entities):
		train = train[~train[e].isin(val_entity_instances[i])]

	return {'train': train.reset_index(drop = True),
			'valid': val.reset_index(drop = True),
			'test': test.reset_index(drop = True)}


def create_scaffold_split(df, seed, frac, entity):
	"""create scaffold split. it first generates molecular scaffold for each molecule and then split based on scaffolds
	reference: https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py

	Args:
	    df (pd.DataFrame): dataset dataframe
	    fold_seed (int): the random seed
	    frac (list): a list of train/valid/test fractions
	    entity (str): the column name for where molecule stores
	
	Returns:
	    dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""
	
	try:
		from rdkit import Chem
		from rdkit.Chem.Scaffolds import MurckoScaffold
		from rdkit import RDLogger
		RDLogger.DisableLog('rdApp.*')
	except:
		raise ImportError("Please install rdkit by 'conda install -c conda-forge rdkit'! ")
	from tqdm import tqdm
	from random import Random

	from collections import defaultdict
	random = Random(seed)

	s = df[entity].values
	scaffolds = defaultdict(set)
	idx2mol = dict(zip(list(range(len(s))),s))

	error_smiles = 0
	for i, smiles in tqdm(enumerate(s), total=len(s)):
		try:
			scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol = Chem.MolFromSmiles(smiles), includeChirality = False)
			scaffolds[scaffold].add(i)
		except:
			print_sys(smiles + ' returns RDKit error and is thus omitted...')
			error_smiles += 1

	train, val, test = [], [], []
	train_size = int((len(df) - error_smiles) * frac[0])
	val_size = int((len(df) - error_smiles) * frac[1])
	test_size = (len(df) - error_smiles) - train_size - val_size
	train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

	#index_sets = sorted(list(scaffolds.values()), key=lambda i: len(i), reverse=True)
	index_sets = list(scaffolds.values())
	big_index_sets = []
	small_index_sets = []
	for index_set in index_sets:
		if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
			big_index_sets.append(index_set)
		else:
			small_index_sets.append(index_set)
	random.seed(seed)
	random.shuffle(big_index_sets)
	random.shuffle(small_index_sets)
	index_sets = big_index_sets + small_index_sets

	if frac[2] == 0:
		for index_set in index_sets:
			if len(train) + len(index_set) <= train_size:
				train += index_set
				train_scaffold_count += 1
			else:
				val += index_set
				val_scaffold_count += 1
	else:
		for index_set in index_sets:
			if len(train) + len(index_set) <= train_size:
				train += index_set
				train_scaffold_count += 1
			elif len(val) + len(index_set) <= val_size:
				val += index_set
				val_scaffold_count += 1
			else:
				test += index_set
				test_scaffold_count += 1
	
	train = get_label_weight(train, label_weight_name)
	val = get_label_weight(val, label_weight_name)
	test = get_label_weight(test, label_weight_name)

	return {'train': df.iloc[train].reset_index(drop = True),
			'valid': df.iloc[val].reset_index(drop = True),
			'test': df.iloc[test].reset_index(drop = True)}

def create_combination_split(df, seed, frac):
	"""
	Function for splitting drug combination dataset such that no combinations are shared across the split
	
	Args:
	    df (pd.Dataframe): dataset to split
	    seed (int): random seed
	    frac (list): split fraction as a list
	
	Returns:
	    dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""

	test_size = int(len(df) * frac[2])
	train_size = int(len(df) * frac[0])
	val_size = len(df) - train_size - test_size
	np.random.seed(seed)

	# Create a new column for combination names
	df['concat'] = df['Drug1_ID'] + ',' + df['Drug2_ID']

	# Identify shared drug combinations across all target classes
	combinations = []
	for c in df['Cell_Line_ID'].unique():
		df_cell = df[df['Cell_Line_ID'] == c]
		combinations.append(set(df_cell['concat'].values))

	intxn = combinations[0]
	for c in combinations:
		intxn = intxn.intersection(c)

	# Split combinations into train, val and test
	test_choices = np.random.choice(list(intxn),
						int(test_size / len(df['Cell_Line_ID'].unique())),
						replace=False)
	trainval_intxn = intxn.difference(test_choices)
	val_choices = np.random.choice(list(trainval_intxn),
						int(val_size / len(df['Cell_Line_ID'].unique())),
						replace=False)

	## Create train and test set
	test_set = df[df['concat'].isin(test_choices)].drop(columns=['concat'])
	val_set = df[df['concat'].isin(val_choices)]
	train_set = df[~df['concat'].isin(test_choices)].reset_index(drop=True)
	train_set = train_set[~train_set['concat'].isin(val_choices)]

	return {'train': train_set.reset_index(drop = True),
			'valid': val_set.reset_index(drop = True),
			'test': test_set.reset_index(drop = True)}

# create time split

def create_fold_time(df, frac, date_column):
	"""create splits based on time
	
	Args:
	    df (pd.DataFrame): the dataset dataframe
	    frac (list): list of train/valid/test fractions
	    date_column (str): the name of the column that contains the time info
	
	Returns:
	    dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""
	df = df.sort_values(by = date_column).reset_index(drop = True)
	train_frac, val_frac, test_frac = frac[0], frac[1], frac[2]

	split_date = df[:int(len(df) * (train_frac + val_frac))].iloc[-1][date_column]
	test = df[df[date_column] >= split_date].reset_index(drop = True)
	train_val = df[df[date_column] < split_date]

	split_date_valid = train_val[:int(len(train_val) * train_frac/(train_frac + val_frac))].iloc[-1][date_column]
	train = train_val[train_val[date_column] <= split_date_valid].reset_index(drop = True)
	valid = train_val[train_val[date_column] > split_date_valid].reset_index(drop = True)

	return {'train': train, 'valid': valid, 'test': test, 'split_time': {'train_time_frame': (df.iloc[0][date_column], split_date_valid), 
                                                                         'valid_time_frame': (split_date_valid, split_date), 
                                                                         'test_time_frame': (split_date, df.iloc[-1][date_column])}}



def create_group_split(train_val, seed, holdout_frac, group_column):
	"""split within each stratification defined by the group column for training/validation split
	
	Args:
	    train_val (pd.DataFrame): the train+valid dataframe to split on
	    seed (int): the random seed
	    holdout_frac (float): the fraction of validation
	    group_column (str): the name of the group column
	
	Returns:
	    dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
	"""
	train_df = pd.DataFrame()
	val_df = pd.DataFrame()

	for i in train_val[group_column].unique():
		train_val_temp = train_val[train_val[group_column] == i]
		np.random.seed(seed)
		msk = np.random.rand(len(train_val_temp)) < (1 - holdout_frac)
		train_df = train_df.append(train_val_temp[msk])
		val_df = val_df.append(train_val_temp[~msk])

	return {'train': train_df.reset_index(drop = True), 'valid': val_df.reset_index(drop = True)}
