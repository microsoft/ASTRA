"""
Code for self-training with weak rules.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
from collections import defaultdict
implemented_metrics = ['acc', 'prec', 'rec', 'f1', 'weighted_acc', 'weighted_f1']

class Evaluator:
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.metric = args.metric
        assert self.metric in implemented_metrics, "Evaluation metric not implemented: {}".format(self.metric)

    def evaluate(self, preds, labels, proba=None, comment="", verbose=True):
        assert len(preds) == len(labels), "pred should have same length as true: pred={} gt={}".format(
            len(preds),
            len(labels)
        )

        preds = np.array(preds)
        labels = np.array(labels)

        total_num = len(preds)
        self.logger.info("Evaluating {} on {} examples".format(comment, total_num))

        # Ignore pred == -1 but also penalize by considering all of them as wrong predictions...
        ignore_ind = preds == -1
        keep_ind = preds != -1
        ignore_num = np.sum(ignore_ind)
        ignore_perc = ignore_num / float(total_num)
        if ignore_num > 0:
            self.logger.info("Ignoring {:.4f}% ({}/{}) predictions".format(100*ignore_perc, ignore_num, total_num))

        preds = preds[keep_ind]
        labels = labels[keep_ind]
        if proba is not None:
            proba = proba[keep_ind]
        if len(preds) == 0:
            self.logger.info("Passed empty {} list to Evaluator. Skipping evaluation".format(comment))
            return defaultdict(int)

        pred = list(preds)
        true = list(labels)
        acc = accuracy_score(y_true=true, y_pred=pred)
        f1 = f1_score(y_true=true, y_pred=pred, average='macro')
        prec, rec, fscore, support = precision_recall_fscore_support(y_true=true, y_pred=pred, average='macro')
        conf_mat = confusion_matrix(y_true=true, y_pred=pred)
        clf_report = classification_report(y_true=true, y_pred=pred)

        weighted_acc, weighted_f1 = compute_weighted_acc_f1(y_true=true, y_pred=pred)
        adjust_coef = (total_num - ignore_num) / float(total_num)

        res = {
            'acc': 100 * acc * adjust_coef,
            'weighted_acc': 100 * weighted_acc * adjust_coef,
            'prec': 100 * prec * adjust_coef,
            'rec': 100 * rec * adjust_coef,
            'f1': 100 * f1 * adjust_coef,
            'weighted_f1': 100 * weighted_f1 * adjust_coef,
            'ignored': ignore_num,
            'total': total_num
        }
        res["perf"] = res[self.metric]

        self.logger.info("{} performance: {:.2f}".format(comment, res["perf"]))
        if verbose:
            self.logger.info("{} confusion matrix:\n{}".format(comment, conf_mat))
            self.logger.info("{} report:\n{}".format(comment, clf_report))

        return res


def compute_weighted_acc_f1(y_true, y_pred):
    prec, rec, fscore, support = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
    weighted_acc = np.sum(support * rec) / np.sum(support)
    weighted_f1 = np.mean(fscore)
    return weighted_acc, weighted_f1