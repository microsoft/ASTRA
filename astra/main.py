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
from copy import deepcopy
from collections import defaultdict
from utils import to_one_hot, evaluate_ran, analyze_rule_attention_scores, evaluate, evaluate_test, save_and_report_results
home = expanduser("~")


def wst(args, logger):
    """
        Self-training with weak supervivsion
        Leverages labeled, unlabeled data and weak rules for training a neural network
    """

    teacher_dev_res_list = []
    teacher_test_res_list = []
    teacher_train_res_list = []
    dev_res_list = []
    test_res_list = []
    train_res_list = []
    results = {}

    student_pred_list = []

    ev = Evaluator(args, logger=logger)

    logger.info("building student: {}".format(args.student_name))
    student = Student(args, logger=logger)

    logger.info("building teacher")
    teacher = Teacher(args, logger=logger)

    logger.info("loading data")
    dh = DataHandler(args, logger=logger, student_preprocess=student.preprocess, teacher_preprocess=teacher.preprocess)
    train_dataset = dh.load_dataset(method='train')
    train_dataset.oversample(args.oversample)  
    dev_dataset = dh.load_dataset(method='dev')
    test_dataset = dh.load_dataset(method='test')
    unlabeled_dataset = dh.load_dataset(method='unlabeled')

    logger.info("creating pseudo-dataset")
    pseudodataset = dh.create_pseudodataset(unlabeled_dataset)
    pseudodataset.downsample(args.sample_size)

    # Train Student
    newtraindataset = dh.create_pseudodataset(train_dataset)
    newtraindataset.balance('labels')
    newtraindataset.report_stats('labels')
    results['student_train'] = student.train(
        train_dataset=newtraindataset,
        dev_dataset=dev_dataset,
        train_label_name='labels',
        dev_label_name='labels',
        eval_fn=ev.evaluate
    )
    train_res_list.append(results['student_train'])
    student.save('supervised_student')

    logger.info("\n\n\t*** Evaluating on dev data ***")
    results['supervised_student_dev'] = evaluate(student, dev_dataset, ev, "student dev")
    dev_res_list.append(results['supervised_student_dev'])

    logger.info("\n\n\t*** Evaluating on test data ***")
    results['supervised_student_test'], s_test_dict = evaluate_test(student, test_dataset, ev, "student test")
    test_res_list.append(results['supervised_student_test'])
    student_pred_list.append(s_test_dict)

    # Initialize Teacher
    logger.info("initializing teacher on unlabeled data with majority voting")
    teacher_res = teacher.train(pseudodataset)

    logger.info("evaluating majority voting")
    results['teacher_train'] = evaluate(teacher, train_dataset, ev, "teacher train")
    results['teacher_dev'] = evaluate(teacher, dev_dataset, ev, "teacher dev")
    results['teacher_test'] = evaluate(teacher, test_dataset, ev, "teacher test")
    teacher_train_res_list.append(results['teacher_train'])
    teacher_dev_res_list.append(results['teacher_dev'])
    teacher_test_res_list.append(results['teacher_test'])

    # Self-Training with Weak Supervision
    for iter in range(args.num_iter):
        logger.info("\n\n\t *** Starting loop {} ***".format(iter))

        # Create pseudo-labeled dataset
        pseudodataset.downsample(args.sample_size)

        # Add Student as extra rule in teacher.
        logger.info("Adding Student as extra rule in Teacher")
        teacher.student = student

        _ = teacher.train_ran(train_dataset=train_dataset, train_label_name='labels',
                              dev_dataset=dev_dataset, dev_label_name='labels',
                              unlabeled_dataset=pseudodataset)

        # Apply Teacher on unlabeled data
        teacher_pred_dict_unlabeled = teacher.predict_ran(dataset=pseudodataset)
        teacher_dev_res, t_dev_dict = evaluate_ran(teacher, dev_dataset, ev, "teacher dev iter{}".format(iter))
        teacher_dev_res_list.append(teacher_dev_res)

        teacher_test_res, t_test_dict = evaluate_ran(teacher, test_dataset, ev, "teacher test iter{}".format(iter))
        analyze_rule_attention_scores(t_test_dict, logger, args.logdir, name='test_iter{}'.format(iter), verbose=True)
        teacher_test_res_list.append(teacher_test_res)

        # Update unlabeled data with Teacher's predictions
        pseudodataset.data['teacher_labels'] = teacher_pred_dict_unlabeled['preds']
        pseudodataset.data['teacher_proba'] = teacher_pred_dict_unlabeled['proba']
        pseudodataset.data['teacher_weights'] = np.max(teacher_pred_dict_unlabeled['proba'], axis=1)
        pseudodataset.drop(col='teacher_labels', value=-1)

        if args.uniform_balance:
            pseudodataset.balance('teacher_labels', proba='teacher_proba')
        elif args.uniform_balance_maxsize:
            pseudodataset.balance('teacher_labels', proba='teacher_proba', max_size=args.max_size)
    
        pseudodataset.report_stats('teacher_labels')

        if len(set(teacher_pred_dict_unlabeled['preds'])) == 1:
            logger.info("Self-training led to trivial predictions. Stopping...")
            break

        if len(pseudodataset) < 5:
            logger.info("[WARNING] Sampling led to only {} examples. Skipping iteration...".format(len(pseudodataset)))
            continue

        # Re-train student with weighted pseudo-instances
        logger.info('training student on pseudo-labeled instances provided by the teacher')
        train_res = student.train_pseudo(
            train_dataset=pseudodataset,
            dev_dataset=dev_dataset,
            train_label_name='teacher_proba' if args.soft_labels else 'teacher_labels',
            train_weight_name='teacher_weights' if args.loss_weights else None,
            dev_label_name='labels',
            eval_fn=ev.evaluate
        )

        if not args.nofinetune:
            logger.info('fine-tuning the student on clean labeled data')
            train_res = student.finetune(
                train_dataset=newtraindataset,
                dev_dataset=dev_dataset,
                train_label_name='labels',
                dev_label_name='labels',
                eval_fn=ev.evaluate
            )
        train_res_list.append(train_res)

        # Evaluate student performance and update records
        dev_res = evaluate(student, dev_dataset, ev, "student dev iter{}".format(iter))
        test_res, s_test_dict = evaluate_test(student, test_dataset, ev, "student test iter{}".format(iter))
        logger.info("Student Dev performance on iter {}: {}".format(iter, dev_res['perf']))
        logger.info("Student Test performance on iter {}: {}".format(iter, test_res['perf']))

        prev_max = max([x['perf'] for x in dev_res_list])
        if dev_res['perf'] > prev_max:
            logger.info("Improved dev performance from {:.2f} to {:.2f}".format(prev_max, dev_res['perf']))
            student.save("student_best")
            teacher.save("teacher_best")
        dev_res_list.append(dev_res)
        test_res_list.append(test_res)
        student_pred_list.append(s_test_dict)

    # Store Final Results
    logger.info("Final Results")
    teacher_all_dev = [x['perf'] for x in teacher_dev_res_list]
    teacher_all_test = [x['perf'] for x in teacher_test_res_list]
    teacher_perf_str = ["{}:\t{:.2f}\t{:.2f}".format(i, teacher_all_dev[i], teacher_all_test[i]) for i in np.arange(len(teacher_all_dev))]
    logger.info("TEACHER PERFORMANCES:\n{}".format("\n".join(teacher_perf_str)))

    all_dev = [x['perf'] for x in dev_res_list]
    all_test = [x['perf'] for x in test_res_list]
    perf_str = ["{}:\t{:.2f}\t{:.2f}".format(i, all_dev[i], all_test[i]) for i in np.arange(len(all_dev))]
    logger.info("STUDENT PERFORMANCES:\n{}".format("\n".join(perf_str)))

    # Get results in the best epoch (if multiple best epochs keep last one)
    best_dev_epoch = len(all_dev) - np.argmax(all_dev[::-1]) - 1
    best_test_epoch = len(all_test) - np.argmax(all_test[::-1]) - 1
    logger.info("BEST DEV {} = {:.3f} for epoch {}".format(args.metric, all_dev[best_dev_epoch], best_dev_epoch))
    logger.info("FINAL TEST {} = {:.3f} for epoch {} (max={:.2f} for epoch {})".format(args.metric,
                                                                                       all_test[best_dev_epoch], best_dev_epoch, all_test[best_test_epoch], best_test_epoch))
    results['teacher_train_iter'] = teacher_train_res_list
    results['teacher_dev_iter'] = teacher_dev_res_list
    results['teacher_test_iter'] = teacher_test_res_list

    results['student_train_iter'] = train_res_list
    results['student_dev_iter'] = dev_res_list
    results['student_test_iter'] = test_res_list

    results['student_dev'] = dev_res_list[best_dev_epoch]
    results['student_test'] = test_res_list[best_dev_epoch]
    results['teacher_dev'] = teacher_dev_res_list[best_dev_epoch]
    results['teacher_test'] = teacher_test_res_list[best_dev_epoch]
    
    # Save models and results
    student.save("student_last")
    teacher.save("teacher_last")
    save_and_report_results(args, results, logger)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset name", type=str, default='youtube')
    parser.add_argument("--experiment_folder", help="Dataset name", type=str, default='../experiments/')
    parser.add_argument("--datapath", help="Path to base dataset folder", type=str, default='../data')
    parser.add_argument("--dataset_type", help="Dataset type (original / train_valid_split", type=str, default='train_valid_split')
    parser.add_argument("--rule_perc", default=1.0, type=float, help="Percentage of rules to keep")
    parser.add_argument("--logdir", help="Experiment log directory", type=str, default='./')
    parser.add_argument("--metric", help="Evaluation metric", type=str, default='acc')
    parser.add_argument("--student_name", help="Student short name", type=str, default='berttf')
    parser.add_argument("--teacher_name", help="Student short name", type=str, default='ran_ssl')
    parser.add_argument("--num_iter", help="Number of self/co-training iterations", type=int, default=25)
    parser.add_argument('-ws', '--weak_sources', help="List of weak sources name for Teacher", nargs='+')
    parser.add_argument("--downsample", help="Downsample labeled train & dev datasets randomly stratisfied by label", type=float, default=1.0)
    parser.add_argument("--oversample", help="Oversample labeled train datasets", type=int, default=1)
    parser.add_argument("--tokenizer_method", help="Tokenizer method (for LogReg student)", type=str, default='clean')
    parser.add_argument("--num_epochs", default=70, type=int, help="Total number of training epochs for student.")
    parser.add_argument("--debug", action="store_true", help="Activate debug mode")
    parser.add_argument("--nofinetune", action="store_true", help="Do NOT Fine-tune Student on labeled data after each iteration")
    parser.add_argument("--soft_labels", action="store_true", help="Use soft labels for training Student")
    parser.add_argument("--convert_abstain_to_random", action="store_true", help="In Teacher, if rules abstain on dev/test then flip a coin")
    parser.add_argument("--hard_student_rule", action="store_true", help="When using Student as a rule in Teacher, use hard (instead of soft) student labels")
    parser.add_argument("--balance", action="store_true", help="balance selftraining predictions to retain clean label distribution")
    parser.add_argument("--balance_maxsize", action="store_true", help="balance selftraining predictions to retain clean label distribution")
    parser.add_argument("--uniform_balance", action="store_true", help="balance selftraining predictions to uniform")
    parser.add_argument("--uniform_balance_maxsize", action="store_true", help="balance selftraining predictions to uniform and keep top 1000")
    parser.add_argument("--train_batch_size", help="Train batch size", type=int, default=16) 
    parser.add_argument("--eval_batch_size", help="Dev batch size", type=int, default=128)
    parser.add_argument("--max_size", help="Max size of unlabeled data for training the student if balance_maxsize==True", type=int, default=1000)
    parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA")
    parser.add_argument("--lower_case", action="store_true", help="Use uncased model")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite dataset if exists")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay for student")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--finetuning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--fp16", action='store_true', help='whehter use fp16 or not')
    parser.add_argument("--sample_size", nargs="?", type=int, default=16384, help="number of unlabeled samples for evaluating uncetainty on in each self-training iteration")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    
    args = parser.parse_args()
    if args.seed == 0:
        args.dataset_type = 'original'
    np.random.seed(args.seed)
    
    # Define dataset-specific parameters
    if args.dataset in ['sms']:
        args.num_labels = 2
        args.metric = 'weighted_f1'
        args.max_seq_length = 64
        args.oversample = 3
    elif args.dataset in ['youtube']:
        args.num_labels = 2
        args.metric = 'weighted_acc'
        args.max_seq_length = 64
        args.oversample = 3
    elif args.dataset == 'trec':
        args.num_labels = 6
        args.metric = 'weighted_acc'
        args.max_seq_length = 64
        args.oversample = 10
    elif args.dataset == 'census':
        args.num_labels = 2
        args.metric = 'weighted_acc'
        args.max_seq_length = 64
        args.train_batch_size = 128
        args.oversample = 5
    elif args.dataset == 'mitr':
        args.num_labels = 9
        args.metric = 'weighted_f1'
        args.oversample = 2
        args.max_seq_length = 32
        args.train_batch_size = 256
    elif args.dataset in ['spouse']:
        args.num_labels = 2
        args.metric = 'f1'
        args.max_seq_length = 32
        args.train_batch_size = 256
    else:
        raise(BaseException('unknown dataset: {}'.format(args.dataset)))

    # Start Experiment
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d-%H_%M")

    
    args.experiment_folder = os.path.join(args.experiment_folder, args.dataset)
    args.logdir = os.path.join(args.experiment_folder, args.logdir)
    
    if args.debug:
        args.logdir = os.path.join(args.experiment_folder, 'debug')
        if os.path.exists(args.logdir):
            shutil.rmtree(args.logdir)
    else:
        args.logdir = args.logdir + "/" + date_time
        if args.student_name == 'logreg':
            args.logdir += "_st{}".format(args.student_name.upper())
        else:
            args.logdir += "_{}".format(args.dataset) + \
            "_st{}".format(args.student_name.upper()) + \
            "_epoch{}".format(args.num_epochs) + \
            "_lr{}".format(args.learning_rate) + \
            "_batch{}".format(args.train_batch_size) + \
            "_maxseq{}".format(args.max_seq_length)

        if int(args.downsample) != 1:
            args.logdir += "_downsample{}".format(args.downsample)

    os.makedirs(args.logdir, exist_ok=True)
    logger = get_logger(logfile=os.path.join(args.logdir, 'log.log'))

    # Setup CUDA, GPU & distributed training
    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        device = torch.device("cuda")
        args.n_gpu = 1
    args.device = device
    args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.train_batch_size * max(1, args.n_gpu)

    logger.info("\n\n\t\t *** NEW EXPERIMENT ***\nargs={}".format(args))
    wst(args, logger=logger)
    close(logger)


if __name__ == "__main__":
    main()