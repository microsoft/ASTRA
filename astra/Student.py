"""
Code for self-training with weak supervision.
Author: Giannis Karamanolakis (gkaraman@cs.columbia.edu)
"""

import os
from model import LogRegTrainer, BertTrainer, DefaultModelTrainer

preprocessed_dataset_list = ['trec', 'youtube', 'sms', 'census', 'mitr']
supported_trainers = {
    'logreg': LogRegTrainer,
    'bert': BertTrainer,
}

class Student:
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.name = args.student_name
        assert self.name in supported_trainers, "Student not supported: <{}>".format(self.name)
        self.trainer_class = supported_trainers[self.name]
        if args.dataset in preprocessed_dataset_list:
            self.trainer = DefaultModelTrainer(args=self.args, logger=self.logger)
        else:
            self.trainer = self.trainer_class(args=self.args, logger=self.logger)
        self.preprocess = self.trainer.preprocess

    def train(self, train_dataset, dev_dataset, train_label_name='label', dev_label_name='label'):
        # Training student for the first time on few labeled data (First iteration of self-training)
        res = self.trainer.train(
            train_texts=train_dataset.data['texts'],
            preprocessed_train_texts=train_dataset.data.get('preprocessed_texts'),
            train_labels=train_dataset.data[train_label_name],
            dev_texts=dev_dataset.data['texts'],
            preprocessed_dev_texts=dev_dataset.data.get('preprocessed_texts'),
            dev_labels=dev_dataset.data[dev_label_name],
        )
        return res

    def train_pseudo(self, train_dataset, dev_dataset, train_label_name='label', train_weight_name='weights',
                     dev_label_name='label'):
        # Fine-tuning student on pseudo-labeled data (provided by the Teacher)
        # Call different function for student model with different hyperparameters: weighted training.
        # Note: if train_weight_name is None, then weights are not used
        res = self.trainer.train_pseudo(
            train_texts=train_dataset.data['texts'],
            preprocessed_train_texts=train_dataset.data.get('preprocessed_texts'),
            train_labels=train_dataset.data[train_label_name],
            train_weights=train_dataset.data.get(train_weight_name),
            dev_texts=dev_dataset.data['texts'],
            preprocessed_dev_texts=dev_dataset.data.get('preprocessed_texts'),
            dev_labels=dev_dataset.data[dev_label_name],
        )
        return res

    def finetune(self, train_dataset, dev_dataset, train_label_name='label', dev_label_name='label'):
        # Fine-tuning student on few labeled data
        # Note: this function is different than train() because of potentially different hyperparameters.
        #       If all hyperparameters are same, you can merge both train() and finetune() into one.  
        res = self.trainer.finetune(
            train_texts=train_dataset.data['texts'],
            preprocessed_train_texts=train_dataset.data.get('preprocessed_texts'),
            train_labels=train_dataset.data[train_label_name],
            dev_texts=dev_dataset.data['texts'],
            preprocessed_dev_texts=dev_dataset.data.get('preprocessed_texts'),
            dev_labels=dev_dataset.data[dev_label_name],
        )
        return res

    def predict(self, dataset):
        res = self.trainer.predict(
            texts=dataset.data['texts'],
            preprocessed_texts=dataset.data.get('preprocessed_texts'),
        )
        assert 'preds' in res and 'proba' in res, "Student Trainer must return 'preds' and 'proba'"
        return res

    def save(self, name='student'):
        savefolder = os.path.join(self.args.logdir, name)
        self.logger.info('Saving {} to {}'.format(name, savefolder))
        os.makedirs(savefolder, exist_ok=True)
        self.trainer.save(savefolder)

    def load(self):
        savefolder = os.path.join(self.args.logdir, 'student')
        if not os.path.exists(savefolder):
            raise(BaseException('Pre-trained student folder does not exist: {}'.format(savefolder)))
        self.trainer.load(savefolder)
