"""
Code for self-training with weak supervision.
Author: Giannis Karamanolakis (gkaraman@cs.columbia.edu)
"""

import os
import math
import random
import numpy as np
from numpy.random import seed
import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, Input, Dropout, Dense, Lambda
from tensorflow.keras.models import Model
from scipy.special import softmax
from bert import bert_tokenization
import bert
from bert.loader import load_stock_weights


class DefaultModelTrainer:
    """
    Student Trainer based on default model architectures for equal comparison with previous approaches
    The Trainer considers pre-computed contextualized embeddings that are already provided with previous benchmarks
    It has to implement: __init__, train, evaluate, save, load
    """

    def __init__(self, args, logger=None):
        self.args = args
        self.dataset = args.dataset
        self.name = '{}_CLF'.format(self.dataset)
        self.logger = logger
        self.manual_seed = args.seed
        self.max_seq_length = args.max_seq_length
        self.datapath = args.datapath
        self.lower_case = True
        self.model_dir = args.logdir
        self.tokenizer=None
        self.learning_rate = args.learning_rate
        self.finetuning_rate = args.finetuning_rate
        self.num_supervised_trials = args.num_supervised_trials
        self.sup_batch_size = args.train_batch_size
        self.sup_epochs = args.num_epochs
        self.unsup_epochs = args.num_unsup_epochs
        self.num_labels = args.num_labels
        self.model = None
        self.gpus = None

    def init(self):
        self.model = construct_model(self.max_seq_length, self.num_labels, dataset=self.dataset)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
        return

    def preprocess(self, texts, preprocessed_texts=None):
        return texts

    def train(self, train_texts, train_labels, dev_texts=None, dev_labels=None, eval_fn=None,
              preprocessed_train_texts=None, preprocessed_dev_texts=None):
        self.logger.info("Class labels: {}".format(self.num_labels))

        x_train = np.array(self.preprocess(train_texts, preprocessed_train_texts))
        y_train = np.array(train_labels)
        x_dev = np.array(self.preprocess(dev_texts, preprocessed_dev_texts))
        y_dev = np.array(dev_labels)

        self.logger.info("X Train Shape " + str(x_train.shape) + ' ' + str(y_train.shape))
        self.logger.info("X Dev Shape " + str(x_dev.shape) + ' ' + str(y_dev.shape))

        model_file = os.path.join(self.model_dir, "supervised_model.h5")
        distributed_res = self.distributed_train(x_train, y_train, x_dev, y_dev, model_file)

        self.model = distributed_res['model']
        if not os.path.exists(model_file):
            self.model.save_weights(model_file)
            print("Supervised model file saved to {}".format(model_file))

        res = {}
        res['dev_loss'] = distributed_res['dev_loss']
        return res

    def train_pseudo(self, train_texts, train_labels, train_weights, dev_texts=None, dev_labels=None, eval_fn=None,
                     preprocessed_train_texts=None, preprocessed_dev_texts=None):
        x_train = np.array(self.preprocess(train_texts, preprocessed_train_texts))
        y_train = np.array(train_labels)
        x_weight = np.array(train_weights) if train_weights is not None else None
        x_dev = np.array(self.preprocess(dev_texts, preprocessed_dev_texts))
        y_dev = np.array(dev_labels)

        if self.gpus is None:
            self.strategy = tf.distribute.MirroredStrategy()
            gpus = self.strategy.num_replicas_in_sync
            self.gpus = gpus

        if self.model is None:
            self.init()

        with self.strategy.scope():
            if y_train.ndim == 2:
                # support soft labels
                self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                   metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])
                y_dev = to_categorical(y_dev)
            else:
                self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

        self.model.fit(
            x=[x_train, np.zeros((len(x_train), self.max_seq_length))],
            y=y_train,
            validation_data=([x_dev, np.zeros((len(x_dev), self.max_seq_length))], y_dev),
            batch_size=32 * self.gpus,
            shuffle=True,
            sample_weight=x_weight,
            epochs=self.unsup_epochs,
            callbacks=[
                create_learning_rate_scheduler(max_learn_rate=self.learning_rate, end_learn_rate=1e-7,
                                               warmup_epoch_count=3, total_epoch_count=self.unsup_epochs),
                K.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        res = {}
        return res

    def finetune(self, train_texts, train_labels, dev_texts=None, dev_labels=None, eval_fn=None,
                 preprocessed_train_texts=None, preprocessed_dev_texts=None):
        # Similar to training but with smaller learning rate
        x_train = np.array(self.preprocess(train_texts, preprocessed_train_texts))
        y_train = np.array(train_labels)
        x_dev = np.array(self.preprocess(dev_texts, preprocessed_dev_texts))
        y_dev = np.array(dev_labels)

        if self.gpus is None:
            self.strategy = tf.distribute.MirroredStrategy()
            gpus = self.strategy.num_replicas_in_sync
            self.gpus = gpus

        with self.strategy.scope():
            if y_train.ndim == 2:
                # support soft labels
                self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                     metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])
                y_dev = to_categorical(y_dev)
            else:
                self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

        self.model.fit(
            x=[x_train, np.zeros((len(x_train), self.max_seq_length))],
            y=y_train,
            validation_data=([x_dev, np.zeros((len(x_dev), self.max_seq_length))], y_dev),
            batch_size=self.sup_batch_size * self.gpus,
            shuffle=True,
            epochs=self.unsup_epochs,
            callbacks=[
                create_learning_rate_scheduler(max_learn_rate=self.finetuning_rate, end_learn_rate=1e-7,
                                               warmup_epoch_count=3, total_epoch_count=self.unsup_epochs),
                K.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        res = {}
        return res

    def distributed_train(self, x_train, y_train, x_dev, y_dev, model_file):
        N_base = self.num_supervised_trials

        self.strategy = tf.distribute.MirroredStrategy()
        gpus = self.strategy.num_replicas_in_sync
        self.gpus = gpus
        print('Number of devices: {}'.format(gpus))

        best_base_model = None
        best_validation_loss = np.inf

        for counter in range(N_base):
            with self.strategy.scope():
                strong_model = construct_model(self.max_seq_length, self.num_labels, dataset=self.dataset)
                strong_model.compile(optimizer=tf.keras.optimizers.Adam(),
                                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
            if os.path.exists(model_file):
                strong_model.load_weights(model_file)
                best_base_model = strong_model
                print("No Training... Pre-trained supervised model loaded from {}".format(model_file))
                break

            if counter == 0:
                print(strong_model.summary())

            print("training supervised model {}/{}".format(counter, N_base))
            strong_model.fit(
                x=[x_train, np.zeros((len(x_train), self.max_seq_length))],
                y=y_train,
                batch_size=self.sup_batch_size * gpus,
                shuffle=True,
                epochs=self.sup_epochs,
                callbacks=[
                    create_learning_rate_scheduler(max_learn_rate=self.learning_rate, end_learn_rate=1e-7, warmup_epoch_count=20,
                                                   total_epoch_count=self.sup_epochs),
                    K.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
                ],
                validation_data=([x_dev, np.zeros((len(x_dev), self.max_seq_length))], y_dev))

            val_loss = strong_model.evaluate([x_dev, np.zeros((len(x_dev), self.max_seq_length))], y_dev)
            print("Validation loss for run {} : {}".format(counter, val_loss))
            if val_loss[0] < best_validation_loss:
                best_base_model = strong_model
                best_validation_loss = val_loss[0]

        strong_model = best_base_model
        res = strong_model.evaluate([x_dev, np.zeros((len(x_dev), self.max_seq_length))], y_dev)
        print("Best validation loss for base model {}: {}".format(best_validation_loss, res))

        return {
            'dev_loss': best_validation_loss,
            'model': strong_model
        }

    def predict(self, texts, batch_size=256, preprocessed_texts=None, prefix=""):
        x_train = np.array(self.preprocess(texts, preprocessed_texts))
        self.logger.info("Predicting labels for {} texts".format(len(texts)))
        y_pred = self.model.predict(
            [x_train, np.zeros((len(x_train), self.max_seq_length))],
            batch_size=batch_size
        )

        # Get student's features
        layer_name = 'first' #last
        desiredOutputs = [self.model.get_layer(layer_name).output]
        newModel = tf.keras.Model(self.model.inputs, desiredOutputs)
        features = newModel([x_train, np.zeros((len(x_train), self.max_seq_length))])
        preds = np.argmax(y_pred, axis=-1).flatten()
        soft_proba = softmax(y_pred, axis=-1)
        return {
            'preds': preds,
            'proba': soft_proba,
            'features': features.numpy()
        }

    def load(self, savefolder):
        self.logger.info("loading student from {}".format(savefolder))
        raise (BaseException('not implemented'))

    def save(self, savefolder):
        model_file = os.path.join(savefolder, "final_model.h5")
        self.logger.info("Saving model at {}".format(model_file))
        self.model.save_weights(model_file)
        return


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):
    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate * math.exp(
                math.log(end_learn_rate / max_learn_rate) * (epoch - warmup_epoch_count + 1) / (
                        total_epoch_count - warmup_epoch_count + 1))
        return float(res)

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler

def construct_model(max_seq_length, num_labels, dense_dropout=0.5, dataset='trec'):
    # Constructing default model architectures for equal comparison with previous approaches
    if dataset == 'trec':
        emb_size = 1024
        hidden_size = 512
        num_layers = 2
    elif dataset == 'youtube':
        emb_size = 16634
        hidden_size = 512
        num_layers = 0
    elif dataset == 'sms':
        emb_size = 1024
        hidden_size = 512
        num_layers = 2
    elif dataset == 'census':
        emb_size = 105
        hidden_size = 256
        num_layers = 2
    elif dataset == 'mitr':
        emb_size = 1024
        hidden_size = 512
        num_layers = 2
    elif dataset in ['spouse']:
        emb_size = 768
        hidden_size = 512
        num_layers = 5
    else:
        raise(BaseException("Default model not available for {}".format(dataset)))

    features = Input(shape=(emb_size,), name="first")
    hidden = Dropout(dense_dropout)(features)

    for i in range(num_layers):
        name = 'dense{}'.format(i) if i != num_layers - 1 else 'last'
        hidden = Dense(units=hidden_size, activation="relu", name=name)(hidden)
        hidden = Dropout(dense_dropout)(hidden)

    logits = hidden
    outputs = Dense(units=num_labels, activation="softmax", name="output_1")(logits)
    model = tf.keras.Model(inputs=features, outputs=outputs)
    return model
