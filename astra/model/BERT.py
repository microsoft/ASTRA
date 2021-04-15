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

class BertTrainer:
    # Trainer Class
    # has to implement: __init__, train, evaluate, save, load
    def __init__(self, args, logger=None):
        self.args = args
        self.name = 'BERT'
        self.logger = logger
        self.manual_seed = args.seed
        self.max_seq_length = args.max_seq_length
        self.datapath = args.datapath
        self.bert_model_file = os.path.join(self.datapath, 'pretrained_models/bert/')
        self.vocab_file = os.path.join(self.bert_model_file, 'vocab.txt')
        self.lower_case = True
        self.learning_rate = args.learning_rate
        self.finetuning_rate = args.finetuning_rate
        self.model_dir = args.logdir
        self.tokenizer = bert_tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.lower_case)
        self.num_supervised_trials = args.num_supervised_trials
        self.sup_batch_size = args.train_batch_size
        self.sup_epochs = args.num_epochs
        self.unsup_epochs = args.num_unsup_epochs
        self.T = args.T

    def preprocess(self, texts, preprocessed_texts=None):
        if preprocessed_texts is not None:
            return preprocessed_texts
        self.logger.info("Pre-processing {} texts...".format(len(texts)))
        indices = generate_sequence_data(texts, self.max_seq_length, self.tokenizer)
        return indices

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

    def train(self, train_texts, train_labels, dev_texts=None, dev_labels=None, eval_fn=None,
              preprocessed_train_texts=None, preprocessed_dev_texts=None):
        self.num_labels = len(set(train_labels))
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
                strong_model = construct_bert(self.bert_model_file, self.max_seq_length, self.num_labels)
                strong_model.compile(optimizer=K.optimizers.Adam(),
                                     loss=K.losses.SparseCategoricalCrossentropy(from_logits=True),
                                     metrics=[K.metrics.SparseCategoricalAccuracy(name="acc")])

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
                    create_learning_rate_scheduler(max_learn_rate=1e-4, end_learn_rate=1e-7, warmup_epoch_count=20,
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
        x_ids = np.array(self.preprocess(texts, preprocessed_texts))
        self.logger.info("Predicting labels for {} texts".format(len(texts)))
        y_pred = self.model.predict(
            [x_ids, np.zeros((len(x_ids), self.max_seq_length))],
            batch_size=batch_size
        )

        layer_name = 'dense'  # last
        desiredOutputs = [self.model.get_layer(layer_name).output]
        newModel = tf.keras.Model(self.model.inputs, desiredOutputs)
        features = newModel.predict(
            [x_ids, np.zeros((len(x_ids), self.max_seq_length))],
            batch_size=batch_size
        )

        preds = np.argmax(y_pred, axis=-1).flatten()
        soft_proba = softmax(y_pred, axis=-1)
        soft_proba = np.squeeze(soft_proba, axis=None)
        features = np.squeeze(features, axis=None)
        return {
            'preds': preds,
            'proba': soft_proba,
            'features': features,
        }

    def load(self, savefolder):
        self.logger.info("loading student from {}".format(savefolder))
        raise(BaseException('not implemented'))

    def save(self, savefolder):
        model_file = os.path.join(savefolder, "final_model.h5")
        self.logger.info("Saving model at {}".format(model_file))
        self.model.save_weights(model_file)
        return


def generate_sequence_data(texts, MAX_SEQUENCE_LENGTH, tokenizer):
    all_ids = []
    for text in texts:
        tokens = tokenizer.tokenize(text.strip())
        if len(tokens) > MAX_SEQUENCE_LENGTH - 2:
            tokens = tokens[:MAX_SEQUENCE_LENGTH - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = input_ids + [0] * (MAX_SEQUENCE_LENGTH - len(tokens))
        all_ids.append(input_ids)
    return all_ids


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

def construct_bert(model_dir, timesteps, classes, dense_dropout=0.5, attention_dropout=0.3, hidden_dropout=0.3,
                   adapter_size=8):
    bert_ckpt_file = os.path.join(model_dir, "bert_model.ckpt")
    bert_config_file = os.path.join(model_dir, "bert_config.json")
    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    bert_model = bert.BertModelLayer.from_params(bert_params, name="bert")


    input_ids = Input(shape=(timesteps,), dtype='int32', name="input_ids_1")
    token_type_ids = Input(shape=(timesteps,), dtype='int32', name="token_type_ids_1")

    dense = Dense(units=768, activation="tanh", name="dense")
    output = bert_model([input_ids, token_type_ids])  # output: [batch_size, max_seq_len, hidden_size]

    print("bert shape", output.shape)
    cls_out = Lambda(lambda seq: seq[:, 0:1, :])(output)
    cls_out = Dropout(dense_dropout)(cls_out)
    logits = dense(cls_out)
    logits = Dropout(dense_dropout)(logits)
    logits = Dense(units=classes, activation="softmax", name="output_1")(logits)

    model = Model(inputs=[input_ids, token_type_ids], outputs=logits)
    model.build(input_shape=(None, timesteps))

    # load the pre-trained model weights
    load_stock_weights(bert_model, bert_ckpt_file)
    return model
