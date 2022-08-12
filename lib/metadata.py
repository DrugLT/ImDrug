# -*- coding: utf-8 -*-
# Author: ImDrug Team
# License: MIT 

"""This file contains all metadata of datasets in DrugLT.
Attributes:
    adme_dataset_names (list): all adme dataset names
    admet_metrics (dict): a dictionary with key the dataset name and value the recommended metric
    admet_splits (dict): a dictionary with key the dataset name and value the recommended split
    catalyst_dataset_names (list): all catalyst dataset names
    category_names (dict): mapping from ML problem (1st tier) to all tasks
    crisproutcome_dataset_names (list): all crispr outcome dataset names
    dataset_list (list): total list of dataset names in ImDrug
    dataset_names (dict): mapping from task name to list of dataset names
    ddi_dataset_names (list): all ddi dataset names
    develop_dataset_names (list): all develop dataset names
    distribution_oracles (list): all distribution learning oracles, i.e. molecule evaluators
    download_oracle_names (list): oracle names that require downloading predictors
    drugres_dataset_names (list): all drugres dataset names
    drugsyn_dataset_names (list): all drugsyn dataset names
    drugsyn_metrics (dict): a dictionary with key the dataset name and value the recommended metric
    drugsyn_splits (dict):  a dictionary with key the dataset name and value the recommended split
    dti_dataset_names (list): all dti dataset names
    dti_dg_metrics (dict): a dictionary with key the dataset name and value the recommended metric
    dti_dg_splits (dict):  a dictionary with key the dataset name and value the recommended split
    evaluator_name (list): list of evaluator names
    forwardsyn_dataset_names (list): all reaction dataset names
    generation_datasets (list): all generation dataset names
    meta_oracle_name (list): list of all meta oracle names
    molgenpaired_dataset_names (list): all molgenpaired dataset names
    mti_dataset_names (list): all mti dataset names
    name2stats (dict): mapping from dataset names to statistics
    name2type (dict): mapping from dataset names to downloaded file format
    oracle2id (dict): mapping from oracle names to dataverse id
    oracle2type (dict): mapping from oracle names to downloaded file format
    receptor2id (dict): mapping from receptor id to dataverse id 
    oracle_names (list): list of all oracle names
    paired_dataset_names (list): all paired dataset names
    ppi_dataset_names (list): all ppi dataset names
    property_names (list): a list of oracles that correspond to some molecular properties
    qm_dataset_names (list): all qm dataset names
    retrosyn_dataset_names (list): all retrosyn dataset names
    single_molecule_dataset_names (list): all molgen dataset names
    synthetic_oracle_name (list): all oracle names for synthesis
    test_multi_pred_dataset_names (list): test multi pred task name
    test_single_pred_dataset_names (list): test single pred task name
    toxicity_dataset_names (list): all toxicity dataset names
    trivial_oracle_names (list): a list of oracle names for trivial oracles
    yield_dataset_names (list): all yield dataset names
"""
####################################
# test cases
test_single_pred_dataset_names = ['test_single_pred']
test_multi_pred_dataset_names = ['test_multi_pred']

# single_pred prediction

toxicity_dataset_names = ['tox21']

adme_dataset_names = ['bbb_martins']

bioact_dataset_names = ['hiv']

qm_dataset_names = ['qm9']

transpos_dataset_names = ['protein_16-4096']

####################################
# multi_pred prediction

dti_dataset_names = [
                     'sbap',
                     'sbap_reg']

ddi_dataset_names = ['drugbank', 'uspto_50k', 'uspto_1k_TPL', 'uspto_500_MT', 'uspto_catalyst']

ppi_dataset_names = []

yield_dataset_names = ['uspto_yields', 'uspto_500_MT']

catalyst_dataset_names = ['uspto_catalyst']

reacttype_dataset_names = ['uspto_50k', 'uspto_1k_TPL', 'uspto_500_MT']

####################################
# generation

retrosyn_dataset_names = ['uspto50k', 'uspto']

forwardsyn_dataset_names = ['uspto']

single_molecule_dataset_names = ['zinc', 'moses', 'chembl', 'chembl_v29']

paired_dataset_names = ['uspto50k', 'uspto']

####################################
# oracles

evaluator_name = ['roc-auc', 'f1', 'pr-auc', 'precision', 'recall', \
				  'accuracy', 'balanced_accuracy', 'mse', 'rmse', 'mae', 'r2', 'micro-f1', 'macro-f1', \
				  'weighted-f1', 'balanced-f1', 'kappa', 'avg-roc-auc', 'rp@k', 'pr@k', 'pcc', 'spearman']

####################################

category_names = {'single_pred': ["Tox",
									"ADME",
									"BioAct",
									"QM",
									"Yields",],
				'multi_pred': ["DTI",
								"PPI",
								"DDI",
								"Catalyst"],
				'generation': ["RetroSyn",
								"Reaction",
								"MolGen"
								]
				}
input_names = {'multi_pred': {"DTI": ['drug', 'protein'],
								"PPI": ['protein', 'protein'],
								"DDI": ['drug', 'drug'],
								"Catalyst": ['drug', 'drug'],
							}	
				}


def get_task2category():
	task2category = {}
	for i, j in category_names.items():
		for x in j:
			task2category[x] = i
	return task2category

dataset_names = {"Tox": toxicity_dataset_names,
				"ADME": adme_dataset_names, 
				'BioAct': bioact_dataset_names,
				"DTI": dti_dataset_names, 
				"PPI": ppi_dataset_names, 
				"DDI": ddi_dataset_names,
				"QM": qm_dataset_names,
				"Yields": yield_dataset_names, 
				"ReactType": reacttype_dataset_names, 
				"Catalyst": catalyst_dataset_names, 
				"test_single_pred": test_single_pred_dataset_names,
				"test_multi_pred": test_multi_pred_dataset_names,
				"Transposition": transpos_dataset_names, # WARNING: cannot be published
				}

dataset_list = []
for i in dataset_names.keys():
    dataset_list = dataset_list + [i.lower() for i in dataset_names[i]]

name2type = {'toxcast': 'tab',
 'tox21': 'tab',
 'bbb_martins': 'tab',
 'hiv': 'tab',
 'drugbank': 'csv',
 'uspto50k': 'tab',
 'qm9': 'csv',
 'uspto_50k': 'csv',
 'uspto_1k_TPL': 'csv',
 'uspto_500_MT': 'csv',
 'uspto_yields': 'csv',
 'uspto_catalyst': 'csv',
 'test_single_pred': 'tab',
 'test_multi_pred': 'tab',
 'protein_16-4096': 'csv',
 'sbap': 'csv',
 'sbap_reg': 'csv'}

name2stats = {
	'bbb_martins': 1975,
	'tox21': 7831,
	'hiv': 41127,
	'qm9': 133885,
	'sbap': 32140,
	'uspto_yields': 853638,
	'uspto-50k': 50016,
	'uspto-500-MT': 143535,
	'uspto-1k-TPL': 445115,
	'drugbank': 191808,
	'uspto_catalyst': 721799,
}

name2imratio = {
	'tox21': 22.51, 
    'bbb_martins': 3.24,
	'hiv': 27.50,
	'qm': 133883, 
	'sbap': 36.77,
	'uspto_50k': 65.78,
	'uspto_1k_TPL': 110.86,
	'uspto_500_MT': 285.06,
	'uspto_catalyst': 3975.86,
	'uspto_yields': 7.59,
	'drugbank': 10124.67
}

metrics = {'LT Classification': ['balanced_accuracy', 'balanced-f1', 'roc-auc'],
		   'Open LT': ['balanced_accuracy', 'balanced-f1', 'roc-auc'],
		   'LT Regression': ['mse', 'mae']}