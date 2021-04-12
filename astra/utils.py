"""
Code for self-training with weak rules.
"""
import argparse
import json
import logging
import os
import glob
from os.path import expanduser
import numpy as np
import random
import shutil
import torch
import joblib
from Logger import get_logger, close
from DataHandler import DataHandler
from Student import Student
from Teacher import Teacher
from Evaluator import Evaluator
from datetime import datetime
import ust_sampler
home = expanduser("~")
from copy import deepcopy
from collections import defaultdict


def to_one_hot(x, num_classes):
    targets = np.array([x]).reshape(-1)
    return np.eye(num_classes)[targets]


def evaluate_ran(model, dataset, evaluator, comment="test"):
    pred_dict = model.predict_ran(dataset=dataset)
    res = evaluator.evaluate(preds=pred_dict['preds'],
                             labels=dataset.data['labels'],
                             proba=pred_dict['proba'],
                             comment=comment)
    return res, pred_dict

def evaluate(model, dataset, evaluator, comment="test"):
    pred_dict = model.predict(dataset=dataset)
    res = evaluator.evaluate(preds=pred_dict['preds'],
                             labels=dataset.data['labels'],
                             proba=pred_dict['proba'],
                             comment=comment)
    return res


def evaluate_test(model, dataset, evaluator, comment="test"):
    pred_dict = model.predict(dataset=dataset)
    res = evaluator.evaluate(preds=pred_dict['preds'],
                             labels=dataset.data['labels'],
                             proba=pred_dict['proba'],
                             comment=comment)
    return res, pred_dict


def save_and_report_results(args, results, logger):
    logger.info("\t*** Final Results ***")
    for res, values in results.items():
        logger.info("\n{}:\t{}".format(res, values))
    savepath = os.path.join(args.logdir, 'results.pkl')
    logger.info('Saving results at {}'.format(savepath))
    joblib.dump(results, savepath)
    txt_savepath = os.path.join(args.logdir, 'results.txt')

    with open(txt_savepath, 'w') as f:
        def pinfo(text):
            logger.info(text)
            f.write(text + "\n")
        pinfo("Dataset: {}".format(args.dataset))
        pinfo("Weak Sources: {}".format(args.weak_sources))
        pinfo("Model: {}\n".format(args.student_name))
        if 'teacher_train' in results:
            pinfo("Teacher Train {}: {:.1f}".format(args.metric, results['teacher_train']['perf']))
        if 'teacher_dev' in results:
            pinfo("Teacher Dev {}: {:.1f}".format(args.metric, results['teacher_dev']['perf']))
        if 'teacher_test' in results:
            pinfo("Teacher Test {}: {:.1f}\n".format(args.metric, results['teacher_test']['perf']))
        if 'student_dev' in results:
            pinfo("Student Dev {}: {:.1f}".format(args.metric, results['student_dev']['perf']))
        if 'student_test' in results:
            pinfo("Student Test {}: {:.1f}".format(args.metric, results['student_test']['perf']))

    args_savepath = os.path.join(args.logdir, 'args.json')
    with open(args_savepath, 'w') as f:
        args.device = -1
        json.dump(vars(args), f)
    logger.info('Saved report at {}'.format(txt_savepath))
    return

def analyze_rule_attention_scores(res, logger, savefolder, name='test', verbose=True):
    if not 'att_scores' in res or res['att_scores'] is None:
        return
    preds = res['preds']
    proba = res['proba']
    att_scores = res['att_scores']
    rule_mask = res['rule_mask']
    rule_ids = [(np.nonzero(x)[0]).tolist() for x in rule_mask]
    rule_att_scores = [att_score_list[:len(rule_ids[i])].tolist() for i, att_score_list in enumerate(att_scores)]
    flat_att_scores = [score for scores in rule_att_scores for score in scores]
    low_scores = [score for score in flat_att_scores if score < 0.5]
    rule2att = defaultdict(list)
    for i, rids in enumerate(rule_ids):
        for j, rid in enumerate(rids):
            rule2att[rid].append(rule_att_scores[i][j])

    if verbose:
        for rule, scores in sorted(rule2att.items()):
            logger.info("Rule {}: lower att scores = {}, higher att scores = {}".format(rule, sorted(scores)[:10], sorted(scores)[-10:]))

        for rule, scores in sorted(rule2att.items()):
            logger.info("Rule {}: min={:.2f}\tmax={:.2f}\tmean={:.2f}\tmedian={:.2f}\tsupport={:.2f}".format(rule, min(scores), max(scores),
                                                                                                   np.average(scores), np.median(scores), len(scores)))
        logger.info("First 100 rules :\n{}".format(rule_ids[:100]))
        logger.info("First 100 rule attention scores:\n{}".format([float("{:.1f}".format(x)) for x in flat_att_scores[:100]]))
        logger.info("Low scores ({}/{})\t{}".format(len(low_scores), len(flat_att_scores), [float("{:.1f}".format(x)) for x in low_scores[:100]]))

    savefolder = os.path.join(savefolder, 'teacher_dump')
    logger.info("Saving attention scores at {}".format(savefolder))
    os.makedirs(savefolder, exist_ok=True)
    savefile = os.path.join(savefolder, 'att_scores_{}.pkl'.format(name))
    joblib.dump(att_scores, savefile)
    savefile = os.path.join(savefolder, 'teacher_preds_{}.pkl'.format(name))
    joblib.dump(preds, savefile)
    savefile = os.path.join(savefolder, 'teacher_proba_{}.pkl'.format(name))
    joblib.dump(proba, savefile)
    savefile = os.path.join(savefolder, 'rule_mask_{}.pkl'.format(name))
    joblib.dump(rule_mask, savefile)
    savefile = os.path.join(savefolder, 'rule2att_dict_{}.pkl'.format(name))
    joblib.dump(rule2att, savefile)
    return

