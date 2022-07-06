import pandas as pd
import numpy as np
import os, sys, json
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import dgllife.utils.mol_to_graph

from utils import fuzzy_search 
# from metadata import evaluator_name, distribution_oracles
from metadata import evaluator_name

try:
	from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score, precision_recall_curve
	from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, cohen_kappa_score
	from sklearn.utils.multiclass import unique_labels
	from scipy.sparse import coo_matrix
except:
	ImportError("Please install sklearn by 'conda install -c anaconda scikit-learn' or 'pip install scikit-learn '! ")

def accuracy(y_true, y_pred):
    cnt = y_true.shape[0]
    true_count = (y_pred == y_true).sum()
    now_accuracy = true_count / cnt
    return now_accuracy, cnt

def avg_auc(y_true, y_pred):
	scores = []
	for i in range(np.array(y_true).shape[0]):
	    scores.append(roc_auc_score(y_true[i], y_pred[i]))
	return sum(scores)/len(scores)

def rmse(y_true, y_pred):
	return np.sqrt(mean_squared_error(y_true, y_pred))

def recall_at_precision_k(y_true, y_pred, threshold = 0.9):
	pr, rc, thr = precision_recall_curve(y_true, y_pred)
	if len(np.where(pr >= threshold)[0]) > 0:
		return rc[np.where(pr >= threshold)[0][0]]
	else:
		return 0.

def precision_at_recall_k(y_true, y_pred, threshold = 0.9):
	pr, rc, thr = precision_recall_curve(y_true, y_pred)	 
	if len(np.where(rc >= threshold)[0]) > 0:
		return pr[np.where(rc >= threshold)[0][-1]]
	else:
		return 0.

def pcc(y_true, y_pred):
		return np.corrcoef(y_true, y_pred)[1,0]

def roc_auc(y_true, y_pred, sample_weight=None, per_class=False):
	from sklearn.utils import check_consistent_length, check_array
	from sklearn.metrics._ranking import _binary_roc_auc_score
	# np.set_printoptions(threshold=sys.maxsize)

	# get one-hot labels
	labels = np.array(y_true)
	onehot_labels = np.zeros((labels.size, int(labels.max()+1)))
	onehot_labels[np.arange(labels.size), labels.astype('int64')] = 1
	
	y_true = onehot_labels
	check_consistent_length(y_true, y_pred, sample_weight)
	y_true = check_array(y_true)
	# try:
	y_pred = check_array(y_pred)
	# except:
	# 	print ('ValueError: Input contains NaN, infinity or a value too large for dtype('float32').')
	# 	y_pred = np.random


	n_classes = y_pred.shape[-1]
	score = np.zeros((n_classes,))
	for c in range(n_classes):
		y_true_c = y_true.take([c], axis=-1).ravel()
		y_pred_c = y_pred.take([c], axis=-1).ravel()
		try:
			score[c] = _binary_roc_auc_score(y_true_c, y_pred_c, sample_weight=sample_weight)
		except:
			score[c] = 0.5
	if per_class:
		return score
	else:
		return np.mean(score)

def balanced_accuracy_score(y_true, y_pred, sample_weight=None, per_class=False):
	if sample_weight is None:
		sample_weight = np.ones_like(y_true)
	y_true, y_pred, sample_weight = np.array(y_true), np.array(y_pred), np.array(sample_weight)
	labels = unique_labels(y_true, y_pred)
	n_labels = labels.size

	# If labels are not consecutive integers starting from zero, then
	# y_true and y_pred must be converted into index form
	need_index_conversion = not (
		labels.dtype.kind in {"i", "u", "b"}
		and np.all(labels == np.arange(n_labels))
		and y_true.min() >= 0
		and y_pred.min() >= 0
	)
	if need_index_conversion:
		label_to_ind = {y: x for x, y in enumerate(labels)}
		y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
		y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
	
	# intersect y_pred, y_true with labels, eliminate items not in labels
	ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
	if not np.all(ind):
		y_pred = y_pred[ind]
		y_true = y_true[ind]
		# also eliminate weights of eliminated items
		sample_weight = sample_weight[ind]

	# Choose the accumulator dtype to always have high precision
	if sample_weight.dtype.kind in {"i", "u", "b"}:
		dtype = np.int64
	else:
		dtype = np.float64

	cm = coo_matrix(
		(sample_weight, (y_true, y_pred)),
		shape=(n_labels, n_labels),
		dtype=dtype,
	).toarray()

	fusion_matrix = ConfusionMatrix(n_labels)
	fusion_matrix.matrix = cm

	with np.errstate(divide="ignore", invalid="ignore"):
		rec_per_class = np.diag(cm) / cm.sum(axis=1)
	if np.any(np.isnan(per_class)):
		warnings.warn("y_pred contains classes not in y_true")
		rec_per_class = rec_per_class[~np.isnan(per_class)]
	if per_class:
		return rec_per_class

	score = np.nanmean(rec_per_class)
	return score

def balanced_f1(y_true, y_pred, sample_weight=None, per_class=False):
	"""
	Assuming i.i.d of the samples in each class, to compensate for the effect of imbalanced distribution on f1 score, we calculate
	precision based on false negatives weighted by the support of each ground truth class.
	"""

	if sample_weight is None:
		sample_weight = np.ones_like(y_true)

	y_true, y_pred, sample_weight = np.array(y_true), np.array(y_pred), np.array(sample_weight)
	labels = unique_labels(y_true, y_pred)
	n_labels = labels.size

	# If labels are not consecutive integers starting from zero, then
	# y_true and y_pred must be converted into index form
	need_index_conversion = not (
		labels.dtype.kind in {"i", "u", "b"}
		and np.all(labels == np.arange(n_labels))
		and y_true.min() >= 0
		and y_pred.min() >= 0
	)
	if need_index_conversion:
		label_to_ind = {y: x for x, y in enumerate(labels)}
		y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
		y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

	# intersect y_pred, y_true with labels, eliminate items not in labels
	ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
	if not np.all(ind):
		y_pred = y_pred[ind]
		y_true = y_true[ind]
		# also eliminate weights of eliminated items
		sample_weight = sample_weight[ind]

	# Choose the accumulator dtype to always have high precision
	if sample_weight.dtype.kind in {"i", "u", "b"}:
		dtype = np.int64
	else:
		dtype = np.float64

	cm = coo_matrix(
		(sample_weight, (y_true, y_pred)),
		shape=(n_labels, n_labels),
		dtype=dtype,
	).toarray()

	fusion_matrix = ConfusionMatrix(n_labels)
	fusion_matrix.matrix = cm

	rec_per_class = fusion_matrix.get_rec_per_class()
	prec_adjusted = []
	for i in range(fusion_matrix.num_classes):
		weight = fusion_matrix.matrix[i, :].sum()/np.sum(fusion_matrix.matrix, axis=1)
		weight[np.isnan(weight)] = 0
		weight[np.isinf(weight)] = 0
		prec_adjusted.append(fusion_matrix.matrix[i, i] / (fusion_matrix.matrix[:, i] * weight).sum())
	prec_per_class = np.array(prec_adjusted)
	balanced_f1 = 2 * (prec_per_class * rec_per_class) / (prec_per_class + rec_per_class)
	balanced_f1[np.isnan(balanced_f1)] = 0
	if per_class:
		return balanced_f1
	return np.mean(balanced_f1)

class Evaluator:

	"""evaluator to evaluate predictions
	
	Args:
		name (str): the name of the evaluator function
	"""
	
	def __init__(self, name):
		"""create an evaluate object
		"""
		self.name = fuzzy_search(name, evaluator_name)
		self.assign_evaluator()

	def assign_evaluator(self):
		"""obtain evaluator function given the evaluator name
		"""
		if self.name == 'roc-auc':
			# from functools import partial
			# self.evaluator_func = partial(roc_auc_score, average=None)  
			self.evaluator_func = roc_auc
		elif self.name == 'f1':
			self.evaluator_func = f1_score 
		elif self.name == 'pr-auc':
			self.evaluator_func = average_precision_score 
		elif self.name == 'rp@k':
			self.evaluator_func = recall_at_precision_k
		elif self.name == 'pr@k':
			self.evaluator_func = precision_at_recall_k
		elif self.name == 'precision':
			self.evaluator_func = precision_score
		elif self.name == 'recall':
			self.evaluator_func = recall_score
		elif self.name == 'accuracy':
			self.evaluator_func = accuracy_score
		elif self.name == 'balanced_accuracy':
			self.evaluator_func = balanced_accuracy_score
		elif self.name == 'mse':
			self.evaluator_func = mean_squared_error
		elif self.name == 'rmse':
			self.evaluator_func = rmse
		elif self.name == 'mae':
			self.evaluator_func = mean_absolute_error
		elif self.name == 'r2':
			self.evaluator_func = r2_score
		elif self.name == 'pcc':
			self.evaluator_func = pcc
		elif self.name == 'spearman':
			try:
				from scipy import stats
			except:
				ImportError("Please install scipy by 'pip install scipy'! ")
			self.evaluator_func = stats.spearmanr
		elif self.name == 'micro-f1':
			self.evaluator_func = f1_score
		elif self.name == 'macro-f1':
			self.evaluator_func = f1_score
		elif self.name == 'weighted-f1':
			self.evaluator_func = f1_score
		elif self.name == 'balanced-f1':
			self.evaluator_func = balanced_f1
		elif self.name == 'kappa':
			self.evaluator_func = cohen_kappa_score
		elif self.name == 'avg-roc-auc':
			self.evaluator_func = avg_auc
		elif self.name == 'novelty':   	
			from .chem_utils import novelty
			self.evaluator_func = novelty  
		elif self.name == 'diversity':
			from .chem_utils import diversity
			self.evaluator_func = diversity 
		elif self.name == 'validity':
			from .chem_utils import validity
			self.evaluator_func = validity 
		elif self.name == 'uniqueness':
			from .chem_utils import uniqueness
			self.evaluator_func = uniqueness 
		elif self.name == 'kl_divergence':
			from .chem_utils import kl_divergence
			self.evaluator_func = kl_divergence
		elif self.name == 'fcd_distance':
			from .chem_utils import fcd_distance
			self.evaluator_func = fcd_distance

	def __call__(self, *args, **kwargs):
		"""call the evaluator function on targets and predictions
		
		Args:
		    *args: targets, predictions, and other information
		    **kwargs: other auxilliary inputs for some evaluators
		
		Returns:
		    float: the evaluator output
		"""
		# if self.name in distribution_oracles:  
		# 	return self.evaluator_func(*args, **kwargs)	
		# 	#### evaluator for distribution learning, e.g., diversity, validity   
		y_true = kwargs['y_true'] if 'y_true' in kwargs else args[0]
		y_pred = kwargs['y_pred'] if 'y_pred' in kwargs else args[1]
		if len(args)<=2 and 'threshold' not in kwargs:
			threshold = 0.5 
		else:
			threshold = kwargs['threshold'] if 'threshold' in kwargs else args[2]

		### original __call__(self, y_true, y_pred, threshold = 0.5)
		y_true = np.array(y_true)
		y_pred = np.array(y_pred)
		if self.name in ['precision','recall','f1','accuracy']:
			y_pred = [1 if i > threshold else 0 for i in y_pred]
		if self.name in ['micro-f1', 'macro-f1', 'weighted-f1']:
			return self.evaluator_func(y_true, y_pred, average = self.name.split('-')[0])
		if self.name in ['rp@k', 'pr@k']:
			return self.evaluator_func(y_true, y_pred, threshold = threshold)
		if self.name == 'spearman':
			return self.evaluator_func(y_true, y_pred)[0]
		# return self.evaluator_func(y_true, y_pred)
		return self.evaluator_func(*args, **kwargs)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, label, output):
        length = output.shape[0]
        for i in range(length):
            self.matrix[label[i], output[i]] += 1

    def get_rec_per_class(self):
        rec = np.array(
            [
                self.matrix[i, i] / self.matrix[i, :].sum()
                for i in range(self.num_classes)
            ]
        )
        rec[np.isnan(rec)] = 0
        return rec

    def get_pre_per_class(self):
        pre = np.array(
            [
                self.matrix[i, i] / self.matrix[:, i].sum()
                for i in range(self.num_classes)
            ]
        )
        pre[np.isnan(pre)] = 0
        return pre

    def get_accuracy(self):
        acc = (
            np.sum([self.matrix[i, i] for i in range(self.num_classes)])
            / self.matrix.sum()
        )
        return acc


    def plot_confusion_matrix(self, normalize = False, cmap=plt.cm.Blues):
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = self.matrix.T

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=np.arange(self.num_classes), yticklabels=np.arange(self.num_classes),
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        #Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[j, i], fmt),
                        ha="center", va="center",
                        color="white" if cm[j, i] > thresh else "black")
        fig.tight_layout()
        return fig

if __name__ == "__main__":
	y_true = [0, 0, 0, 1, 2, 2, 2, 2, 2, 2]
	y_pred = [0, 0, 1, 1, 0, 1, 1, 2, 2, 2]
	print(balanced_f1(y_pred=y_pred, y_true=y_true))