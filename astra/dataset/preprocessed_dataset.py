"""
Code for self-training with weak supervision.
Author: Giannis Karamanolakis (gkaraman@cs.columbia.edu)
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import glob
import joblib
from copy import deepcopy
import shutil

# PreprocessedDataset is used for loading exactly the same dataset splits and features as in our experiments

class PreprocessedDataset:
    # Load pre-processed dataset as used in https://github.com/awasthiabhijeet/Learning-From-Rules
    def __init__(self, datapath="../../data", orig_train=True, dataset='trec', seed=42):
        self.dataset = dataset
        self.seed = seed
        self.basedatafolder = os.path.join(datapath, self.dataset.upper())
        self.datafolder = os.path.join(self.basedatafolder, 'seed{}/'.format(seed))
        self.language = 'english'
        self.orig_train = orig_train
        self.label2ind = self.get_label2ind()
        self.num_labels = len(self.label2ind)
        # self.lf_names = ["rule_{}".format(i+1) for i in range(15)]
        self.lf_names = None

    def get_label2ind(self):
        if self.dataset == 'trec':
            return  {"DESC": 0, "ENTY": 1, "HUM": 2, "ABBR": 3, "LOC": 4, "NUM": 5}
        elif self.dataset == 'youtube':
            return {"ham": 0, "spam": 1}
        elif self.dataset == 'sms':
            return {"ham": 0, "spam": 1}
        elif self.dataset == 'census':
            return {"low": 0, "high": 1}
        elif self.dataset == 'mitr':
            return {'O': 0,
                          'Location': 1,
                          'Hours': 2,
                          'Amenity': 3,
                          'Price': 4,
                          'Cuisine': 5,
                          'Dish': 6,
                          'Restaurant_Name': 7,
                          'Rating': 8}
        else:
            raise(BaseException('Pre-trained dataset not supported: {}'.format(self.dataset)))

    def load_data(self, method):
        if method == 'train' and not self.orig_train:
            method = 'unlabeled'
        texts = joblib.load(os.path.join(self.datafolder, "{}_x.pkl".format(method)))
        texts = texts.tolist()

        labels = joblib.load(os.path.join(self.datafolder, "{}_labels.pkl".format(method)))
        labels = labels.squeeze().tolist()
        rule_preds = joblib.load(os.path.join(self.datafolder, "{}_rule_preds.pkl".format(method)))
        rule_preds[rule_preds == self.num_labels] = -1
        rule_preds = rule_preds.tolist()

        if method =='train':
            exemplars = joblib.load(os.path.join(self.datafolder, 'train_exemplars.pkl'))
            return {'texts': texts, 'labels': labels, 'weak_labels': rule_preds, 'exemplar_labels': exemplars}
        else:
            return {'texts': texts, 'labels': labels, 'weak_labels': rule_preds}

    def preprocess(self):
        seed = self.seed
        self.train_valid_split()
        datafolder = os.path.join(self.basedatafolder, 'original_data'.format(seed))
        if not os.path.exists(datafolder):
            raise(BaseException('You need to download the original dataset for pre-processing, otherwise use our splits'))
        savefolder = os.path.join(self.basedatafolder, 'preprocessed_datasets/seed{}/preprocessed/'.format(seed))
        self.preprocess_fn(datafolder=datafolder, savefolder=savefolder)

    def preprocess_fn(self, datafolder, savefolder):
        # Our function used for pre-processing the original datasets and saving them into different splits (for robustness)
        # Original datasets can be found here: https://github.com/awasthiabhijeet/Learning-From-Rules
        # You can alternatively download the pre-processed versions used in our experiments
        # seed0 contains the original dataset. seed<i> (i=1..5) contains 5 random splits of the unlabeled/train/dev data
        # All splits consider the same test set, for a fair comparison to previous approaches.

        os.makedirs(savefolder, exist_ok=True)

        print("\nunlabeled")
        data = load_data(os.path.join(datafolder, 'U_processed.p'))

        print('x: {}'.format(data.x.shape))
        print('rule_preds: {}'.format(data.l.shape))
        joblib.dump(data.x, os.path.join(savefolder, 'unlabeled_x.pkl'))
        joblib.dump(data.L, os.path.join(savefolder, 'unlabeled_labels.pkl'))
        joblib.dump(data.l, os.path.join(savefolder, 'unlabeled_rule_preds.pkl'))


        print("\ntrain")
        data = load_data(os.path.join(datafolder, 'd_processed.p'))
        print('x: {}'.format(data.x.shape))
        print('rule_preds: {}'.format(data.l.shape))
        joblib.dump(data.x, os.path.join(savefolder, 'train_x.pkl'))
        joblib.dump(data.L, os.path.join(savefolder, 'train_labels.pkl'))
        joblib.dump(data.l, os.path.join(savefolder, 'train_rule_preds.pkl'))
        joblib.dump(data.r, os.path.join(savefolder, 'train_exemplars.pkl'))

        print("\nvalidation")
        data = load_data(os.path.join(datafolder, 'validation_processed.p'))
        print('x: {}'.format(data.x.shape))
        print('rule_preds: {}'.format(data.l.shape))
        joblib.dump(data.x, os.path.join(savefolder, 'dev_x.pkl'))
        joblib.dump(data.L, os.path.join(savefolder, 'dev_labels.pkl'))
        joblib.dump(data.l, os.path.join(savefolder, 'dev_rule_preds.pkl'))

        print("\ntest")
        data = load_data(os.path.join(datafolder, 'test_processed.p'))
        print('x: {}'.format(data.x.shape))
        print('rule_preds: {}'.format(data.l.shape))
        joblib.dump(data.x, os.path.join(savefolder, 'test_x.pkl'))
        joblib.dump(data.L, os.path.join(savefolder, 'test_labels.pkl'))
        joblib.dump(data.l, os.path.join(savefolder, 'test_rule_preds.pkl'))
        print('\nsaved files at {}'.format(savefolder))

    def train_valid_split(self):
        seed = self.seed
        np.random.seed(seed)
        datafolder = self.basedatafolder
        savefolder = os.path.join(self.basedatafolder, 'preprocessed_datasets/seed{}/p'.format(seed))
        os.makedirs(savefolder, exist_ok=True)

        print("\nunlabeled")
        unlabeled = load_data(os.path.join(datafolder, 'U_processed.p'))
        train = load_data(os.path.join(datafolder, 'd_processed.p'))
        dev = load_data(os.path.join(datafolder, 'validation_processed.p'))
        test = load_data(os.path.join(datafolder, 'test_processed.p'))

        # concatenate unlabeled, train, and dev datasets
        all = concatenate_data(unlabeled, train)
        all = concatenate_data(all, dev)
        all_ids = ['unlabeled'] * unlabeled.x.shape[0] + ['train'] * train.x.shape[0] +  ['dev'] * dev.x.shape[0]

        # split datasets
        df = pd.DataFrame()
        df['index'] = np.arange(all.x.shape[0])
        df['label'] = all.L
        df['method'] = all_ids

        train_new_df, df = train_test_split(df, train_size=train.x.shape[0], random_state=seed, shuffle=True, stratify=df['label'])
        dev_new_df, unlabeled_new_df = train_test_split(df, train_size=dev.x.shape[0], random_state=seed, shuffle=True, stratify=df['label'])

        train_new = keep_ind(all, train_new_df['index'].to_numpy())
        dev_new = keep_ind(all, dev_new_df['index'].to_numpy())
        unlabeled_new = keep_ind(all, unlabeled_new_df['index'].to_numpy())
        unlabeled_new = discard_r(unlabeled_new)

        assert train_new.x.shape == train.x.shape
        assert dev_new.x.shape == dev.x.shape
        assert unlabeled_new.x.shape == unlabeled.x.shape

        print('Writing new pre-processed (.p) files to {}'.format(savefolder))
        dump_data(os.path.join(savefolder, 'd_processed.p'), train_new)
        dump_data(os.path.join(savefolder, 'validation_processed.p'), dev_new)
        dump_data(os.path.join(savefolder, 'U_processed.p'), unlabeled_new)
        dump_data(os.path.join(savefolder, 'test_processed.p'), test)
        return



# pre-processing code from https://github.com/awasthiabhijeet/Learning-From-Rules
import pickle
import numpy as np
import collections

f_d = 'f_d'
f_d_U = 'f_d_U'
test_w = 'test_w'

train_modes = [f_d, f_d_U]

F_d_U_Data = collections.namedtuple('GMMDataF_d_U', 'x l m L d r')


def discard_r(data):
    r = np.zeros(data.r.shape)
    return F_d_U_Data(data.x, data.l, data.m, data.L, data.d, r)

def concatenate_data(d1, d2):
    x = np.vstack([d1.x, d2.x])
    l = np.vstack([d1.l, d2.l])
    m = np.vstack([d1.m, d2.m])
    L = np.vstack([d1.L, d2.L])
    d = np.vstack([d1.d, d2.d])
    r = np.vstack([d1.r, d2.r])
    return F_d_U_Data(x, l, m, L, d, r)

def keep_ind(data, inds):
    x = data.x[inds]
    l = data.l[inds]
    m = data.m[inds]
    L = data.L[inds]
    d = data.d[inds]
    r = data.r[inds]
    return F_d_U_Data(x, l, m, L, d, r)

def load_data(fname, num_load=None):
    print('Loading from ', fname)
    with open(fname, 'rb') as f:
        x = pickle.load(f)
        l = pickle.load(f).astype(np.int32)
        m = pickle.load(f).astype(np.int32)
        L = pickle.load(f).astype(np.int32)
        d = pickle.load(f).astype(np.int32)
        r = pickle.load(f).astype(np.int32)

        len_x = len(x)
        assert len(l) == len_x
        assert len(m) == len_x
        assert len(L) == len_x
        assert len(d) == len_x
        assert len(r) == len_x

        L = np.reshape(L, (L.shape[0], 1))
        d = np.reshape(d, (d.shape[0], 1))

        if num_load is not None and num_load < len_x:
            x = x[:num_load]
            l = l[:num_load]
            m = m[:num_load]
            L = L[:num_load]
            d = d[:num_load]
            r = r[:num_load]

        return F_d_U_Data(x, l, m, L, d, r)


def dump_data(save_filename, data):
    save_file = open(save_filename, 'wb')
    pickle.dump(data.x, save_file)
    pickle.dump(data.l, save_file)
    pickle.dump(data.m, save_file)
    pickle.dump(data.L, save_file)
    pickle.dump(data.d, save_file)
    pickle.dump(data.r, save_file)
    save_file.close()

