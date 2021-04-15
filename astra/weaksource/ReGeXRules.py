"""
Code for self-training with weak supervision.
Author: Giannis Karamanolakis (gkaraman@cs.columbia.edu)
"""

import numpy as np

# Weak Source Classes
# Here is the place to define heuristic rules (labeling functions)
# Note: most rules are already provided in benchmarks as pre-processed files (for efficiency).

class YoutubeRules:
    # Weak Source Class
    # has to implement apply function that applied to a dataset
    # predict() function that applies to a single text.
    def __init__(self, datapath="../data"):
        self.num_labels = 2
        self.num_rules = 10
        self.preprocess = None

    def apply(self, dataset):
        preds = dataset.data['weak_labels']
        return preds

class SMSRules:
    # Weak Source Class
    # has to implement apply function that applied to a dataset
    # predict() function that applies to a single text.
    def __init__(self, datapath="../data"):

        self.num_labels = 2
        self.num_rules = 73
        self.preprocess = None

    def apply(self, dataset):
        preds = dataset.data['weak_labels']
        return preds

class TRECRules:
    # Weak Source Class
    # has to implement apply function that applied to a dataset
    # predict() function that applies to a single text.
    def __init__(self, datapath="../data"):
        self.num_labels = 6
        self.num_rules = 68
        self.preprocess = None

    def apply(self, dataset):
        preds = dataset.data['weak_labels']
        return preds


class CENSUSRules:
    # Weak Source Class
    # has to implement apply function that applied to a dataset
    # predict() function that applies to a single text.
    def __init__(self, datapath="../data"):
        self.num_labels = 2
        # self.rules = self.load_rules(self.rule_fpath)
        self.num_rules = 83
        self.preprocess = None

    def apply(self, dataset):
        preds = dataset.data['weak_labels']
        return preds


class MITRRules:
    # Weak Source Class
    # has to implement apply function that applied to a dataset
    # predict() function that applies to a single text.
    def __init__(self, datapath="../data"):
        self.num_labels = 9
        # self.rules = self.load_rules(self.rule_fpath)
        #self.num_rules = 15
        self.add_other_rule = True
        self.num_rules = 15 if not self.add_other_rule else 16
        self.preprocess = None

    def apply(self, dataset):
        preds = dataset.data['weak_labels']
        if not self.add_other_rule:
            return preds

        # "other" rule predicts 0 if all the other rules predict -1.
        pred_mat = np.array(preds)
        other_rule = ((pred_mat!=-1).sum(axis=1) != 0).astype(int) * (-1)
        new_preds = np.append(pred_mat, np.expand_dims(other_rule, 1), axis=1)
        return new_preds.tolist()


class SPOUSERules:
    # Weak Source Class
    # has to implement apply function that applied to a dataset
    # predict() function that applies to a single text.
    def __init__(self, datapath="../data"):
        self.num_labels = 2
        self.num_rules = 9
        self.preprocess = None

    def apply(self, dataset):
        preds = dataset.data['weak_labels']
        return preds
