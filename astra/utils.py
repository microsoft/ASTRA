"""
Code for self-training with weak supervision.
Author: Giannis Karamanolakis (gkaraman@cs.columbia.edu)
"""

import json
import os
import glob
import numpy as np
import joblib
from collections import defaultdict
#import matplotlib.pyplot as plt

# Help functions used in main.py


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


def get_results(resfolder):
    params_file = os.path.join(resfolder, 'args.json')
    with open(params_file, 'r') as f:
        data = json.load(f)
        metric = data['metric']

    results_file = os.path.join(resfolder, 'results.pkl')
    res = joblib.load(results_file)

    if 'teacher_dev_iter' in res:
        teacher_all_dev = [x['perf'] for x in res['teacher_dev_iter']]
        teacher_all_test = [x['perf'] for x in res['teacher_test_iter']]
        teacher_all_dev_acc = [x['acc'] for x in res['teacher_dev_iter']]
        teacher_all_test_acc = [x['acc'] for x in res['teacher_test_iter']]
        teacher_perf_str = ["{}:\t{:.2f}\t{:.2f}".format(i, teacher_all_dev[i], teacher_all_test[i]) for i in np.arange(len(teacher_all_dev))]
        print("TEACHER performance for each iteration:\n{}".format("\n".join(teacher_perf_str)))
    else:
        teacher_all_dev, teacher_all_test, teacher_all_dev_acc, teacher_all_test_acc, teacher_perf_str = [0], [0], [0], [0], ''

    student_all_dev = [x['perf'] for x in res['student_dev_iter']]
    student_all_test = [x['perf'] for x in res['student_test_iter']]
    student_all_dev_acc = [x['acc'] for x in res['student_dev_iter']]
    student_all_test_acc = [x['acc'] for x in res['student_test_iter']]
    student_perf_str = ["{}:\t{:.2f}\t{:.2f}".format(i, student_all_dev[i], student_all_test[i]) for i in np.arange(len(student_all_dev))]
    print("STUDENT performance for each iteration:\n{}".format("\n".join(student_perf_str)))
    return {
        'teacher_dev_perf': teacher_all_dev,
        'teacher_test_perf': teacher_all_test,
        'student_dev_perf': student_all_dev,
        'student_test_perf': student_all_test,
        'teacher_dev_acc': teacher_all_dev_acc,
        'teacher_test_acc': teacher_all_test_acc,
        'student_dev_acc': student_all_dev_acc,
        'student_test_acc': student_all_test_acc,
        'metric': metric
    }


def summarize_results(basefolder, dataset):
    seedfolders = glob.glob(basefolder + '/*seed*')
    print("parsing {} seed folders".format(len(seedfolders)))
    savefile = os.path.join(basefolder, 'all_res.txt')
    with open(savefile, 'w') as f:
        def pinfo(text):
            print(text)
            f.write(text + "\n")

        all_teacher_perf = []
        all_student_perf = []
        all_teacher_dev_perf = []
        all_student_dev_perf = []
        best_students = []
        best_teachers = []
        student_iters = []
        teacher_iters = []
        best_students_acc = []
        best_teachers_acc = []
        for seedfolder in seedfolders:
            print(seedfolder)
            resfolders = glob.glob(seedfolder + '/*')
            if len(resfolders) > 1:
                print('found more than 1 folders.. multiple experiments..')
                print(resfolders)
                return "multiple folders"

            resfolder = resfolders[0]
            res = get_results(resfolder)
            all_teacher_perf.append(res['teacher_test_perf'])
            all_student_perf.append(res['student_test_perf'])
            all_teacher_perf.append(res['teacher_test_perf'])
            all_student_perf.append(res['student_test_perf'])
            all_teacher_dev_perf.append(res['teacher_dev_perf'])
            all_student_dev_perf.append(res['student_dev_perf'])
            metric = res['metric']

            print("results for {}".format(seedfolder.split('/')[-1]))
            best_student_iter = np.argmax(res['student_dev_perf'])
            best_teacher_iter = np.argmax(res['teacher_dev_perf'])
            best_student = res['student_test_perf'][best_student_iter]
            best_teacher = res['teacher_test_perf'][best_teacher_iter]
            pinfo("Best Student: {:.2f} (iter={})".format(best_student, best_student_iter))
            pinfo("Best Teacher: {:.2f} (iter={})".format(best_teacher, best_teacher_iter))
            best_students.append(best_student)
            best_teachers.append(best_teacher)
            best_students_acc.append(res['student_test_acc'][best_student_iter])
            best_teachers_acc.append(res['teacher_test_acc'][best_teacher_iter])
            student_iters.append(best_student_iter)
            teacher_iters.append(best_teacher_iter)

        avg_teacher_perf = np.average(all_teacher_perf, axis=0)
        std_teacher_perf = np.std(all_teacher_perf, axis=0)
        avg_student_perf = np.average(all_student_perf, axis=0)
        std_student_perf = np.std(all_student_perf, axis=0)

        avg_teacher_dev_perf = np.average(all_teacher_dev_perf, axis=0)
        # std_teacher_dev_perf = np.std(all_teacher_dev_perf, axis=0)
        avg_student_dev_perf = np.average(all_student_dev_perf, axis=0)
        # std_student_dev_perf = np.std(all_student_dev_perf, axis=0)

        num_iter = len(avg_student_perf)
        # iters = np.arange(num_iter)

        best_student_iter = np.argmax(avg_student_dev_perf)
        best_teacher_iter = np.argmax(avg_teacher_dev_perf)
        pinfo("\n\nAverage results across splits")


        if 'acc' in metric:
            pinfo("Best Student: {:.2f} (std={:.2f}, iter={})".format(np.average(best_students), np.std(best_students),
                                                                      int(np.average(student_iters))))
            pinfo("Best Teacher: {:.2f} (std={:.2f}, iter={})".format(np.average(best_teachers), np.std(best_teachers),
                                                                      best_teacher_iter,
                                                                      int(np.average(teacher_iters))))
        else:
            pinfo("Best Student: {:.2f} (std={:.2f}, iter={}), acc={:.2f}, std={:.2f}".format(np.average(best_students),
                                                                                              np.std(best_students),
                                                                                              int(np.average(
                                                                                                  student_iters)),
                                                                                              np.average(
                                                                                                  best_students_acc),
                                                                                              np.std(
                                                                                                  best_students_acc)))
            pinfo("Best Teacher: {:.2f} (std={:.2f}, iter={}), acc={:.2f}, std={:.2f}".format(np.average(best_teachers),
                                                                                              np.std(best_teachers),
                                                                                              int(np.average(
                                                                                                  teacher_iters)),
                                                                                              np.average(
                                                                                                  best_teachers_acc),
                                                                                              np.std(
                                                                                                  best_teachers_acc)))
        """
        # plot performance across iterations
        plt.figure()
        plt.title(dataset)
        if len(avg_teacher_perf) > 1:
            plt.errorbar(iters, avg_teacher_perf, std_teacher_perf, linestyle='-', marker='^', label='teacher')
            plt.fill_between(iters, avg_teacher_perf - std_teacher_perf, avg_teacher_perf + std_teacher_perf,
                             facecolor='#F0F8FF', edgecolor='#8F94CC', alpha=1.0)

        plt.errorbar(iters, avg_student_perf, std_student_perf, linestyle='-', marker='^', label='student')
        plt.fill_between(iters, avg_student_perf - std_student_perf, avg_student_perf + std_student_perf,
                         facecolor='#F0F8FF', edgecolor='#bc5a45', alpha=1.0)

        plt.legend(loc='lower right')
        plt.xlabel('iter')
        plt.ylabel(metric)
        plt.show()
        """
        print('avg_student_perf["{}"] = [{}]'.format(dataset, ', '.join(['{:.2f}'.format(x) for x in avg_student_perf])))
        print('std_student_perf["{}"] = [{}]'.format(dataset, ', '.join(['{:.2f}'.format(x) for x in std_student_perf])))
        print('\n\navg_teacher_perf["{}"] = [{}]'.format(dataset, ', '.join(['{:.2f}'.format(x) for x in avg_teacher_perf])))
        print('std_teacher_perf["{}"] = [{}]'.format(dataset, ', '.join(['{:.2f}'.format(x) for x in std_teacher_perf])))

    print("results stored at {}".format(savefile))
    return {
        'student_perf': avg_student_perf[best_student_iter],
        'student_std': std_student_perf[best_student_iter],
        'teacher_perf': avg_teacher_perf[best_teacher_iter],
        'teacher_std': std_teacher_perf[best_teacher_iter],
    }




def analyze_rule_attention_scores(res, logger, savefolder, name='test', verbose=False):
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
