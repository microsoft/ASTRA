"""
Code for self-training with weak rules.
"""

import argparse
import json
import logging
import os
import random
import joblib
import numpy as np
import multiprocessing as mp
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from Tokenizer import Tokenizer

def identity_fn(doc):
    return doc


class LogRegTrainer:
    # Trainer Class
    # has to implement: __init__, train, evaluate, save, load
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.tokenizer_method = args.tokenizer_method
        self.remove_stopwords = True
        self.vocab_path = os.path.join(args.experiment_folder, "preprocessed/")
        self.tokenizer = None
        self.seed = args.seed
        self.tokenizer_obj = Tokenizer(language='english',
                                       tokenizer_method=self.tokenizer_method,
                                       remove_stopwords=self.remove_stopwords,
                                       ngram_range=(1, 1),
                                       min_freq=1,
                                       max_freq_perc=1.0)
        self.tokenizer = self.tokenizer_obj.tokenizer
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.9, norm='l2',
                                          ngram_range=(1, 2), analyzer='word', tokenizer=identity_fn,
                                          preprocessor=identity_fn,
                                          token_pattern=None)
        self.model = LogisticRegression(random_state=self.seed, max_iter=int(1e6))
        self.finetune = self.train_pseudo
    
    def preprocess(self, texts, preprocessed_texts=None):
        """
        Pre-processes a list of texts into lists of tokenized texts
        :param texts: input list of texts
        :param preprocessed_texts: already pre-processed list of texts
        :return: tokenized texts
        """
        if preprocessed_texts is not None:
            return preprocessed_texts
        self.logger.info("tokenizing {} documents".format(len(texts)))
        with mp.Pool(processes=mp.cpu_count()) as pool:
            tokenized_texts = pool.map(partial(self.tokenizer), texts)
        return tokenized_texts

    def train(self, train_texts, train_labels, dev_texts=None, dev_labels=None, eval_fn=None,
              preprocessed_train_texts=None, preprocessed_dev_texts=None):
        logger = self.logger
        tokenized_texts = self.preprocess(train_texts, preprocessed_train_texts)
        logger.info("Fitting vectorizer on {} texts".format(len(train_texts)))
        features = self.vectorizer.fit_transform(tokenized_texts).toarray()
        logger.info("Training logistic regression: {}".format(features.shape))
        self.model.fit(features, train_labels)
        logger.info("logreg weights: {} ({})".format(self.model.coef_.shape[1] * self.model.coef_.shape[0],
                                                     self.model.coef_.shape))
        res = {}
        return res

    def train_pseudo(self, train_texts, train_labels, train_weights=None, dev_texts=None, dev_labels=None, eval_fn=None,
                     preprocessed_train_texts=None, preprocessed_dev_texts=None):
        logger = self.logger
        tokenized_texts = self.preprocess(train_texts, preprocessed_train_texts)
        logger.info("Fitting vectorizer on {} texts".format(len(train_texts)))
        features = self.vectorizer.fit_transform(tokenized_texts).toarray()
        train_labels = np.array(train_labels)
        if train_labels.ndim == 2:
            train_labels = np.argmax(train_labels, axis=-1)
        if train_weights is not None:
            train_weights = np.array(train_weights)
            logger.info("Training logistic regression: {}\nFirst Weights:{}".format(features.shape, train_weights[:10]))
        else:
            logger.info("Training logistic regression: {}\nFirst Weights: None".format(features.shape))
        self.model.fit(features, train_labels, sample_weight=train_weights)
        logger.info("logreg weights: {} ({})".format(self.model.coef_.shape[1] * self.model.coef_.shape[0],
                                                     self.model.coef_.shape))
        res = {}
        return res

    def predict(self, texts, preprocessed_texts=None, prefix=""):
        logger = self.logger
        tokenized_texts = self.preprocess(texts, preprocessed_texts)
        features = self.vectorizer.transform(tokenized_texts).toarray()
        logger.info("predicting labels: {}".format(features.shape))
        preds = self.model.predict(features)
        soft_proba = self.model.predict_proba(features)
        res = {
            'preds': preds,
            'proba': soft_proba,
            'features': features
        }
        return res

    def save(self, savefolder):
        self.logger.info("saving student at {}".format(savefolder))
        joblib.dump(self.tokenizer_obj, os.path.join(savefolder, 'tokenizer_obj.pkl'))
        joblib.dump(self.vectorizer, os.path.join(savefolder, 'vectorizer.pkl'))
        joblib.dump(self.model, os.path.join(savefolder, 'logreg.pkl'))

    def load(self, savefolder):
        self.logger.info("loading student from {}".format(savefolder))
        self.tokenizer_obj = joblib.load(os.path.join(savefolder, 'tokenizer_obj.pkl'))
        self.tokenizer = self.tokenizer_obj.tokenizer
        self.vectorizer = joblib.load(os.path.join(savefolder, 'vectorizer.pkl'))
        self.model = joblib.load(os.path.join(savefolder, 'logreg.pkl'))

        