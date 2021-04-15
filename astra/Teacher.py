"""
Code for self-training with weak supervision.
Author: Giannis Karamanolakis (gkaraman@cs.columbia.edu)
"""

import os
import numpy as np
import random
from weaksource import SMSRules, TRECRules, YoutubeRules, CENSUSRules, MITRRules, SPOUSERules
from RuleAttentionNetwork import RAN

supported_weak_sources = {
    'smsrules': SMSRules,
    'trecrules': TRECRules,
    'youtuberules': YoutubeRules,
    'censusrules': CENSUSRules,
    'mitrrules': MITRRules,
    'spouserules': SPOUSERules,
}

class Teacher:
    """
    Teacher:
        (1) considers multiple weak sources (1) multiple weak (heuristic) rules, (2) Student
        (2) aggregates weak sources with an aggregation model (e.g., RAN) to compute a single pseudo-label
    """

    def __init__(self, args, logger=None):
        self.name = args.teacher_name
        self.datapath = args.datapath
        if self.name != "ran":
            raise (BaseException("Teacher not supported: {}".format(self.name)))
        if args.weak_sources is None:
            if args.dataset in ['sms', 'trec', 'youtube', 'census', 'mitr', 'spouse']:
                args.weak_sources = ["{}rules".format(args.dataset)]
            else:
                raise (BaseException("Teacher not available for dataset={}".format(args.dataset)))
            logger.info("No weak sources specified for Teacher. Using default: {}".format(args.weak_sources))
        else:
            logger.info("weak sources: {}".format(args.weak_sources))
        self.args = args
        self.logger = logger
        self.seed = args.seed
        self.num_labels = self.args.num_labels
        np.random.seed(self.seed)
        self.source_names = args.weak_sources
        print(self.source_names)
        for source_name in self.source_names:
            assert source_name in supported_weak_sources, "Weak Source not supported: {}".format(source_name)
        self.weak_sources = {src: supported_weak_sources[src](self.datapath) for src in self.source_names}
        self.num_rules = np.sum([src.num_rules for _, src in self.weak_sources.items()])
        self.preprocess_fns = [src.preprocess for src_name, src in self.weak_sources.items()]
        self.preprocess = None if None in self.preprocess_fns else self.preprocess_all

        self.agg_model = RAN(args=self.args, num_rules=self.num_rules, logger=self.logger, name=self.name)
        self.name = 'ran'
        self.student = None
        self.convert_abstain_to_random = args.convert_abstain_to_random

    def preprocess_all(self, dataset):
        all_preds = []
        for src_name, src in self.weak_sources.items():
            preds = src.preprocess(dataset)
            all_preds.append(preds)
        if len(all_preds) > 1:
            raise(BaseException('pre-processing not implemented for multiple sources yet...'))
        return all_preds[0]

    def apply(self, dataset):
        # Apply Teacher on unlabeled data
        all_preds = []
        for src_name, weak_src in self.weak_sources.items():
            # Each source is a set of rules.
            num_rules = weak_src.num_rules
            self.logger.info("Applying Teacher with {} LF(s) on {} data".format(num_rules, len(dataset)))

            # preds: num_examples x num_rules
            preds = weak_src.apply(dataset)
            preds = np.array(preds).astype(int)
            if preds.ndim == 1:
                # make sure arrays are 2D
                preds = preds[..., np.newaxis]
            all_preds.append(preds)

        # all_preds: num_examples x num_rules
        all_preds = np.hstack(all_preds)
        return all_preds

    def train(self, dataset):
        weak_labels = self.apply(dataset)
        res = self.aggregate_sources(weak_labels, train=True)
        return {
            "preds": res['preds'],
            "proba": res['proba'],
            "lf_weights": res['lf_weights']
        }

    def predict(self, dataset, student_features=None):
        weak_labels = self.apply(dataset)
        res = self.aggregate_sources(weak_labels, student_features, train=False)
        if dataset.method in ['test', 'dev'] and self.convert_abstain_to_random:
            aggregated_labels = [x if x != -1 else np.random.choice(np.arange(self.num_labels), 1)[0] for x in res['preds'].tolist()]
            res['preds'] = np.array(aggregated_labels)
        return res

    def predict_ran(self, dataset):
        rule_pred = self.apply(dataset)
        student_pred_dict = self.student.predict(dataset=dataset)
        student_pred_proba = student_pred_dict['proba']
        res = self.aggregate_sources(rule_pred,
                                     student_features=student_pred_dict['features'],
                                     student_pred=student_pred_proba,
                                     train=False)
        # self.logger.info("First 10 teacher proba:\n{}".format(res['proba'][:10]))
        if dataset.method in ['test', 'dev'] and self.convert_abstain_to_random:
            labels = [x if x != -1 else np.random.choice(np.arange(self.num_labels), 1)[0] for x in res['preds'].tolist()]
            res['preds'] = np.array(labels)
        return res

    def update(self, train_dataset, train_student_features=None, train_label_name='student_labels',
               dev_dataset=None, dev_student_features=None, dev_label_name='labels',
               unsup_dataset=None, unsup_student_features=None,):
        self.logger.info("Getting rule predictions on train dataset")
        rule_pred_train = self.apply(train_dataset)
        self.logger.info("Getting rule predictions on dev dataset")
        rule_pred_dev = self.apply(dev_dataset)
        if unsup_dataset is not None:
            rule_pred_unsup = self.apply(unsup_dataset)
        else:
            rule_pred_unsup = None
        assert ((rule_pred_train != -1).sum(axis=1) == 0).sum() == 0, "cannot train RAN in examples where no rules apply. need to drop these examples first"
        self.logger.info("Training Rule Attention Network")
        self.agg_model.train(
            x_train=train_student_features,
            rule_pred_train=rule_pred_train,
            y_train=train_dataset.data[train_label_name],
            x_dev=dev_student_features,
            rule_pred_dev=rule_pred_dev,
            y_dev=dev_dataset.data[dev_label_name],
            x_unsup=unsup_student_features,
            rule_pred_unsup=rule_pred_unsup,
        )
        return {}

    def train_ran(self, train_dataset=None, train_label_name='student_labels',
               dev_dataset=None, dev_label_name='labels', unlabeled_dataset=None):
        self.logger.info("Getting rule predictions")
        rule_pred_train = self.apply(train_dataset) if train_dataset is not None else None
        rule_pred_dev = self.apply(dev_dataset) if dev_dataset is not None else None
        rule_pred_unsup = self.apply(unlabeled_dataset) if unlabeled_dataset is not None else None

        self.logger.info("Getting student predictions on train (and dev) dataset")
        assert self.student is not None, "To train RAN we need access to the Student"
        student_pred_train = self.student.predict(dataset=train_dataset) if train_dataset is not None else {'features': None, 'proba': None}
        student_pred_dev = self.student.predict(dataset=dev_dataset) if dev_dataset is not None else {'features': None, 'proba': None}
        student_pred_unsup = self.student.predict(dataset=unlabeled_dataset) if unlabeled_dataset is not None else {'features': None, 'proba': None}

        self.logger.info("Training Rule Attention Network")
        self.agg_model.train(
            x_train=student_pred_train['features'],
            rule_pred_train=rule_pred_train,
            student_pred_train=student_pred_train['proba'],
            y_train=train_dataset.data[train_label_name] if train_dataset is not None else None,
            x_dev=student_pred_dev['features'],
            rule_pred_dev=rule_pred_dev,
            student_pred_dev=student_pred_dev['proba'],
            y_dev=dev_dataset.data[dev_label_name] if dev_dataset is not None else None,
            x_unsup=student_pred_unsup['features'],
            rule_pred_unsup=rule_pred_unsup,
            student_pred_unsup=student_pred_unsup['proba'],
        )
        return {}

    def aggregate_sources(self, weak_labels, student_features=None, train=False, student_pred=None):
        assert weak_labels.shape[1] == self.num_rules, "num rules = {} but weak_labels.shape={}".format(self.num_rules,
                                                                                                    weak_labels.shape[1])
        self.active_rules = np.sum(weak_labels != -1, axis=0) != 0
        self.logger.info("There are {}/{} active rules".format(np.sum(self.active_rules), weak_labels.shape[1]))
        coverage = (np.sum(weak_labels != -1, axis=1) != 0).sum()
        self.logger.info("Coverage: {:.1f}% ({}/{})".format(100*coverage/weak_labels.shape[0], coverage, weak_labels.shape[0]))

        # Train aggregator
        self.lf_weights = None
        if train and self.name == "ran":
            self.agg_model.init(weak_labels)
        elif train:
            raise(BaseException("Teacher method not implemented: {}".format(self.name)))

        if self.lf_weights is not None:
            self.logger.info("Aggregating sources with weights ({}):\n{}".format(self.lf_weights.shape, self.lf_weights))
        
        res = self.agg_model.predict(rule_pred=weak_labels, student_features=student_features, student_pred=student_pred)
        agg_labels = res['preds']
        agg_proba = res['proba']
        att_scores = res['att_scores']
        rule_mask = res['rule_mask']

        return {
            'preds': agg_labels,
            "proba": agg_proba,
            'lf_weights': self.lf_weights,
            'att_scores': att_scores,
            'rule_mask': rule_mask,
        }

    def save(self, savename=None):
        if savename is None:
            savefolder = os.path.join(self.args.logdir, 'teacher')
        else:
            savefolder = os.path.join(self.args.logdir, savename)

        self.logger.info("Saving teacher at {}".format(savefolder))
        os.makedirs(savefolder, exist_ok=True)
        self.agg_model.save(os.path.join(savefolder, 'rule_attention_network.h5'))
        return

    def load(self, savefolder):
        self.logger.info("Loading teacher from {}".format(savefolder))
        self.agg_model.load(os.path.join(savefolder, 'rule_attention_network.h5'))
        
        