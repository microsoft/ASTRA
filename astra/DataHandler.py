"""
Code for self-training with weak rules.
"""

import os
from dataset import PreprocessedDataset
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd
import torch
from itertools import chain
from tqdm.auto import tqdm, trange
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.utils import shuffle

preprocessed_dataset_list = ['trec', 'youtube', 'sms', 'census', 'mitr', 'spouse']

def get_dataset_obj(dataset):
    if dataset in preprocessed_dataset_list:
        return PreprocessedDataset
    else:
        raise(BaseException('dataset not supported: {}'.format(dataset)))


class WSDataset(Dataset):
    def __init__(self, args, method, logger=None, student_preprocess=None, teacher_preprocess=None):
        super(WSDataset, self).__init__()
        self.args = args
        self.seed = args.seed
        self.dataset = args.dataset
        self.datapath = args.datapath
        self.method = method
        self.downsample = args.downsample
        self.savefolder = os.path.join(args.experiment_folder, "preprocessed/")
        os.makedirs(self.savefolder, exist_ok=True)
        self.student_preprocess = student_preprocess
        self.teacher_preprocess = teacher_preprocess
        self.logger = logger
        self.dtype = args.dataset_type
        self.rule_perc = args.rule_perc
        if self.dataset in preprocessed_dataset_list or self.dataset in preprocessed_dataset_list2:
            self.dataset_obj = get_dataset_obj(args.dataset)(datapath=self.datapath, dataset=self.dataset, seed=self.seed, dtype=self.dtype, rule_perc=self.rule_perc)
        else:
            self.dataset_obj = get_dataset_obj(args.dataset)(datapath=self.datapath)
        num_labels = len(self.dataset_obj.label2ind)
        assert args.num_labels == num_labels, "misagreement in #classes: {} vs {}".format(args.num_labels, num_labels)
        self.num_labels = num_labels
        self.data = {}
        self.load_dataset()
        self.label2ind = self.dataset_obj.label2ind
        self.ind2label = {i: label for label, i in self.label2ind.items()}
        if self.method not in ['test', 'unlabeled'] and self.downsample != 1:
            self.downsample_labeled_data()
        if 'labels' in self.data:
            self.label_ratios = [np.sum(self.data['labels'] == label)/float(len(self.data['labels'])) for label in np.arange(self.num_labels)]
        self.report_stats()

    def report_stats(self):
        if 'labels' in self.data:
            df = pd.DataFrame()
            df['ind'] = np.arange(len(self.data['labels']))
            df['label'] = self.data['labels']
            df['label'] = df['label'].map(lambda x: self.ind2label.get(x))
            self.logger.info("{} DATASET: {} examples".format(self.method, df.shape[0]))
            self.logger.info("{} LABELS:\n{}".format(self.method, df['label'].value_counts()))
        return

    def load_dataset(self):
        if not os.path.exists(self.dataset_obj.datafolder):
            self.dataset_obj.preprocess()
        data = self.dataset_obj.load_data(self.method)
        for key, value in data.items():
            self.data[key] = value
        if self.student_preprocess is not None:
            self.logger.info("Pre-processing {} data for student...".format(self.method))
            self.data['preprocessed_texts'] = self.student_preprocess(self.data['texts'])
        if self.teacher_preprocess is not None:
            self.logger.info("Pre-processing {} data for teacher...".format(self.method))
            self.data['preprocessed_lfs'] = self.teacher_preprocess(self)

    def oversample(self, oversample_num):
        # Over-sample labeled data
        # Used for fair comparison to the "ImplyLoss" paper that is doing the same pre-processing step.
        self.logger.info("Oversampling {} data {} times".format(self.method, oversample_num))
        for key, value in self.data.items():
            oversampled_values = []
            for i in range(oversample_num):
                oversampled_values.extend(value)
            self.data[key] = oversampled_values
        self.report_stats()
        return

    def downsample_labeled_data(self):
        downsample = self.downsample
        labeled_size = len(self.data['labels'])
        assert downsample > 0, "downsample must be >0"
        assert 'labels' in self.data, "cannot downsample labeled data as no labeled data are available"
        if downsample >= labeled_size:
            self.logger.info("WARNING! downsample={} > {} labels ({}).".format(downsample, self.method, labeled_size))
            self.logger.info("Not downsampling from the {} set".format(self.method))
            return
        downsample = int(downsample) if downsample > 1 else downsample
        self.logger.info('DOWNSAMPLING: {}'.format(downsample))
        df = pd.DataFrame()
        df['ind'] = np.arange(labeled_size)
        df['label'] = self.data['labels']
        keep, drop = train_test_split(df, train_size=downsample, random_state=self.seed,
                                      shuffle=True, stratify=df['label'])

        self.logger.info('Gold labels after downsampling {} data: {}\n{}'.format(
            self.method,
            keep.shape[0],
            keep['label'].value_counts()))
        keep_indices = keep['ind'].tolist()
        ignore_indices = [i for i in np.arange(labeled_size) if i not in keep_indices]

        # keep original data
        self.original_data = deepcopy(self.data)

        # downsample labeled data
        for key, value in self.original_data.items():
            self.data[key] = [value[i] for i in keep_indices]

        return

    def preprocess_sentiment_weak_labels(self):
        # Apply Weak Sources only once and then just load their results. 
        return

    def __len__(self):
        return len(self.data['texts'])

    def __getitem__(self, item):
        ret = {
            'text': self.data['texts'][item],
            'input_ids': torch.tensor(self.data['input_ids'][item]),
            'attention_mask': torch.tensor(self.data['attention_mask'][item]),
            'label': torch.tensor(self.data['labels'][item]) if 'labels' in self.data else None
        }
        return ret


class PseudoDataset(Dataset):
    def __init__(self, args, wsdataset, logger=None):
        super(PseudoDataset, self).__init__()
        self.args = args
        self.seed = args.seed
        self.dataset = args.dataset
        self.method = wsdataset.method
        self.logger = logger
        self.num_labels = wsdataset.num_labels
        self.dataset_obj = wsdataset.dataset_obj
        self.logger.info("copying data from {} dataset".format(wsdataset.method))
        if args.dataset == 'mitr':
            self.original_data = wsdataset.data
            self.data = self.original_data
        else:
            self.original_data = deepcopy(wsdataset.data)
            self.data = deepcopy(self.original_data)
        self.logger.info("done")
        self.label2ind = wsdataset.label2ind
        self.ind2label = wsdataset.ind2label


    def keep(self, keep_indices, update=None):
        self.logger.info("Creating Pseudo Dataset with {} items...".format(len(keep_indices)))
        for key, values in self.data.items():
            self.data[key] = [values[i] for i in keep_indices]
        self.data['original_indices'] = keep_indices
        if update is not None:
            for key, values in update.items():
                self.logger.info("Updating {}".format(key))
                assert len(values) == len(keep_indices), "update values need to have same dimension as indices {} vs {}".format(len(values), len(keep_indices))
                self.data[key] = list(values)

    def report_stats(self, column_name='labels'):
        if 'labels' in self.data:
            df = pd.DataFrame()
            df['ind'] = np.arange(len(self.data[column_name]))
            df['label'] = self.data[column_name]
            df['label'] = df['label'].map(lambda x: self.ind2label.get(x))
            self.logger.info("PSEUDO-DATASET:\n{} examples\nPSEUDO-LABELS:\n{}".format(df.shape[0], df['label'].value_counts()))
        return

    def downsample(self, sample_size):
        N = len(self.original_data['texts'])
        if sample_size > N:
            self.logger.info("[WARNING] sample size = {} > {}".format(sample_size, N))
            sample_size = N
        self.logger.info("Downsampling {} data".format(sample_size))
        self.data = {}
        keep_indices = np.random.choice(N, sample_size, replace=False)
        for key, values in self.original_data.items():
            self.data[key] = [values[i] for i in keep_indices]

    def drop(self, col='teacher_labels', value=-1):
        indices = [i for i, l in enumerate(self.data[col]) if l != value]
        self.keep(indices)

    def append(self, dataset, merge_cols=None):
        """
        :param dataset: the dataset to append to self
        :param merge_cols: a dictionary showing which columns to merge...
        :return: self + dataset concatenated
        """
        self.logger.info("Merging datasets {}, {}".format(self.method, dataset.method))
        self.logger.info("Size before merging: {}".format(len(self)))
        def extend_values(col1, col2):
            if isinstance(col1, list) and isinstance(col2, list):
                col1.extend(col2)
            elif isinstance(col1, list):
                if col2.ndim == 2:
                    col1 = np.repeat(np.array(col1)[..., np.newaxis], col2.shape[1], axis=1)
                else:
                    col2 = col2.tolist()
                col1 = np.concatenate([col1, col2])
            elif isinstance(col2, list):
                if col1.ndim == 2:
                    col2 = np.repeat(np.array(col2)[..., np.newaxis], col1.shape[1], axis=1)
                else:
                    col1 = col1.tolist()
                col1 = np.concatenate([col1, col2])
            else:
                col1 = np.concatenate([col1, col2])
            return col1

        if merge_cols is None:
            merge_cols = {}

        N = len(self.original_data['texts'])
        M = len(dataset)
        common = set(self.data) & set(dataset.data)
        self_only = set(self.data) - set(dataset.data)
        other_only = set(dataset.data) - set(self.data)

        for key in common:
            #self.data[key].extend(dataset.data[key])
            self.data[key] = extend_values(self.data[key], dataset.data[key])

        for key in self_only:
            #self.data[key].extend([-1] * M)
            self.data[key] = extend_values(self.data[key], [-1] * M)

        for key in other_only:
            self.data[key] = [-1] * N
            # self.data[key].extend(dataset.data[key])
            self.data[key] = extend_values(self.data[key], dataset.data[key])

        for col1, col2 in merge_cols.items():
            self.logger.info("Merging {} to {}".format(col1, col2))
            self.data[col1][-M:] = dataset.data[col2]
        self.logger.info("Size after merging: {}".format(len(self)))
        return

    def balance(self, label_name='labels', label_ratios=None, proba=None, max_size=None, upsample_minority=True):
        # Compute #examples to keep per class
        keep_nums = self.compute_num_examples_to_keep(label_ratios, label_name, max_size, upsample_minority)
        keep_indices = []
        for label in self.ind2label:
            keep_num = keep_nums[label]
            indices = [ind for ind, l in enumerate(self.data[label_name]) if l == label]
            if keep_num > len(indices):
                # Need to oversample dataset: multiply as many times needed to surpass keep_num
                if len(indices) == 0:
                    self.logger.info("ZEROCLASSWARNING: Class {} has zero examples...".format(label))
                    indices = []
                else:
                    indices = indices * int(np.ceil(keep_num / len(indices))) # + np.random.choice(indices, keep_num-len(indices)).tolist()
            if proba is None:
                # keep random samples with this label
                indices = shuffle(indices)
                indices = indices[:keep_num]
            else:
                # keep most confident samples with this label
                keep_proba = np.array(self.data[proba])[indices]
                max_proba = np.max(keep_proba, axis=1)
                indices = np.array(indices)[np.argsort(max_proba)[::-1]][:keep_num].tolist()
                indices = shuffle(indices)
            keep_indices.extend(indices)
        keep_indices = sorted(keep_indices)
        self.logger.info("Balancing Pseudo Dataset to keep {} items...".format(len(keep_indices)))
        for key, values in self.data.items():
            self.data[key] = [values[i] for i in keep_indices]
        return

    def compute_num_examples_to_keep(self, label_ratios, label_name, max_size, upsample_minority=True):
        if label_ratios is None:
            label_ratios = [1/float(self.num_labels) for label in self.ind2label]
        total = len(self) if max_size is None else min(len(self), max_size)
        num_occurs = np.bincount(self.data[label_name], minlength=self.num_labels)

        # Find how to keep balanced classes
        if not upsample_minority:
            ideal_keep_nums = np.array([int(ratio*total) for ratio in label_ratios])
            diff = num_occurs - ideal_keep_nums
            if np.sum(diff < 0) > 0:
                min_class_ind = np.argmin(diff)
                min_occur = max(2, num_occurs[min_class_ind])
                newtotal = np.ceil(min_occur / label_ratios[min_class_ind])
                self.logger.info("[WARNING] Dropping samples from {} to {} to keep balance".format(total, newtotal))
                keep_nums = [int(np.ceil(ratio * newtotal)) for ratio in label_ratios]
            else:
                keep_nums = [int(np.ceil(ratio * total)) for ratio in label_ratios]
        else:
            if max_size is None:
                max_occur = np.max(num_occurs)
                keep_nums = np.array([max_occur for ratio in label_ratios])
            else:
                keep_nums = np.array([int(np.ceil(ratio * total)) for ratio in label_ratios])
        return keep_nums

    def __len__(self):
        return len(self.data['texts'])

    def __getitem__(self, item):
        ret = {
            'text': self.data['texts'][item],
            'input_ids': torch.tensor(self.data['input_ids'][item]),
            'attention_mask': torch.tensor(self.data['attention_mask'][item]),
            'label': torch.tensor(self.data['labels'][item]) if 'labels' in self.data else None
        }
        return ret


class DataHandler:
    # This module is responsible for feeding the data to teacher/student
    # If teacher is applied, then student gets the teacher-labeled data instead of ground-truth labels
    def __init__(self, args, logger=None, student_preprocess=None, teacher_preprocess=None):
        self.args = args
        self.dataset = args.dataset
        self.logger = logger
        self.student_preprocess = student_preprocess
        self.teacher_preprocess = teacher_preprocess
        self.datasets = {}
        self.seed = args.seed
        np.random.seed(self.seed)

    def load_dataset(self, method='train'):
        dataset = WSDataset(self.args, method=method,
                            student_preprocess=self.student_preprocess,
                            teacher_preprocess=self.teacher_preprocess,
                            logger=self.logger)
        self.datasets[method] = dataset
        return dataset

    def create_pseudodataset(self, wsdataset):
        dataset = PseudoDataset(self.args, wsdataset, self.logger)
        return dataset