"""
Code for self-training with weak supervision.
Author: Giannis Karamanolakis (gkaraman@cs.columbia.edu)
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import multiprocessing as mp
from functools import partial
from nltk.corpus import stopwords

# Tokenizer class used only for the Logistic Regression classifier.

def identity_fn(doc):
    return doc


def split_to_sentences(text):
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)


def test_clean_str(text, language='english'):
    """
    Method to pre-process an text for training word embeddings.
    This is post by Sebastian Ruder: https://s3.amazonaws.com/aylien-main/data/multilingual-embeddings/preprocess.py
    and is used at this paper: https://arxiv.org/pdf/1609.02745.pdf
    """
    """
    Cleans an input string and prepares it for tokenization.
    :type text: unicode
    :param text: input text
    :return the cleaned input string
    """
    text = text.lower()

    # replace all numbers with 0
    text = re.sub(r"[-+]?[-/.\d]*[\d]+[:,.\d]*", ' 0 ', text)

    # English-specific pre-processing
    if language == 'english':
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)

    elif language == 'french':
        # French-specific pre-processing
        text = re.sub(r"c\'", " c\' ", text)
        text = re.sub(r"l\'", " l\' ", text)
        text = re.sub(r"j\'", " j\' ", text)
        text = re.sub(r"d\'", " d\' ", text)
        text = re.sub(r"s\'", " s\' ", text)
        text = re.sub(r"n\'", " n\' ", text)
        text = re.sub(r"m\'", " m\' ", text)
        text = re.sub(r"qu\'", " qu\' ", text)

    elif language == 'spanish':
        # Spanish-specific pre-processing
        text = re.sub(r"Â¡", " ", text)

    elif language == 'chinese':
        pass

    text = re.sub(r'[,:;\.\(\)-/"<>]', " ", text)

    # separate exclamation marks and question marks
    text = re.sub(r"!+", " ! ", text)
    text = re.sub(r"\?+", " ? ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class Tokenizer:
    def __init__(self, language, tokenizer_method='spacy', remove_stopwords=True, lowercase=True,
                 strip_accents=None, ngram_range=(1,1), min_freq=1, max_freq_perc=1.0, max_features=None,
                 train_list=None, vocab_loadfolder=None,
                 clean_text=True, vocab_savefolder=None):
        self.language = language
        self.tokenizer_method = tokenizer_method
        self.remove_stopwords = remove_stopwords
        if self.tokenizer_method == 'clean':
            self.tokenize_fn = self.clean_tokenize
            self.tokenizer = self.clean_tokenize
        else:
            raise(Exception("Tokenizer method not implemented: {}".format(self.tokenizer_method)))
        self.stopwords = self.get_stopwords()
        self.ngram_range = ngram_range
        self.min_freq = min_freq
        self.max_freq_perc = max_freq_perc
        self.max_features = max_features
        self.vectorizer = None
        self.lowercase = lowercase
        self.clean_text = clean_text
        self.vocab_loadfolder = vocab_loadfolder
        self.vocab_savefolder = vocab_savefolder
        if strip_accents:
            raise(BaseException("strip accents not supported yet."))
        self.strip_accents = strip_accents
        self.oov = None

        if train_list:
            # If there is a non-empty list of documents then train vocabulary
            print("Training {} vocabulary".format(self.language))
            self.create_vocab(train_list, ngram_range=self.ngram_range, min_freq=self.min_freq,
                              max_freq_perc=self.max_freq_perc, max_features=self.max_features)
            if self.vocab_savefolder:
                self.save_vocab(self.vocab_savefolder)
        elif self.vocab_loadfolder:
            print("Loading pre-trained {} vocabulary from {}".format(self.language, self.vocab_loadfolder))
            self.load_vocab(self.vocab_loadfolder)

    def get_stopwords(self):
        if self.language in ["chinese", "japanese", "catalan", "basque"]:
            return [lex.text.lower() for lex in self.nlp.vocab if lex.is_stop]
        else:
            try:
                return stopwords.words(self.language)
            except:
                return []

    def spacy_tokenize(self, text):
        # clean text
        text = text.strip()

        if self.lowercase:
            text = text.lower()

        if self.clean_text:
            text = self.clean_str(text)

        # tokenize
        tokens = self.tokenizer_obj(text)

        if self.remove_stopwords:
            tokens = [word for word in tokens if not word.is_stop]

        return [token.text for token in tokens]

    def clean_tokenize(self, text):
        text = text.strip()
        text = text.lower()
        text = self.clean_str(text)
        tokens = text.split()
        return tokens

    def clean_str(self, text):
        """
        Method to pre-process an text for training word embeddings.
        This is post by Sebastian Ruder: https://s3.amazonaws.com/aylien-main/data/multilingual-embeddings/preprocess.py
        and is used at this paper: https://arxiv.org/pdf/1609.02745.pdf
        """
        """
        Cleans an input string and prepares it for tokenization.
        :type text: unicode
        :param text: input text
        :return the cleaned input string
        """
        # replace all numbers with 0
        text = re.sub(r"[-+]?[-/.\d]*[\d]+[:,.\d]*", ' 0 ', text)
        text = re.sub(r'[,:;\.\(\)-/"<>]', " ", text)

        # separate exclamation marks and question marks
        text = re.sub(r"!+", " ! ", text)
        text = re.sub(r"\?+", " ? ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


    def get_vectors(self, text):
        return self.vectorizer.transform(text)

    def create_vocab(self, text_list, ngram_range=(1,1), min_freq=3, max_freq_perc=0.9, max_features=None):
        self.vectorizer = CountVectorizer(analyzer='word', tokenizer=identity_fn, preprocessor=identity_fn, token_pattern=None,
                                          ngram_range=ngram_range, min_df=min_freq, max_df=max_freq_perc,
                                          max_features=max_features)  #  strip_accents=self.strip_accents
        with mp.Pool(processes=mp.cpu_count()) as pool:
            tokenized_text = pool.map(partial(self.tokenizer), text_list)
        self.vectorizer = self.vectorizer.fit(tokenized_text)
        self.word2ind = self.vectorizer.vocabulary_
        self.oov = np.max(list(self.word2ind.values()))
        print("{} vocab size: {}".format(self.language, len(self.word2ind)))
        return

    def load_vocab(self, loadfolder):
         self.vectorizer = joblib.load(os.path.join(loadfolder, "{}_vectorizer.pkl".format(self.language)))
         self.word2ind = joblib.load(os.path.join(loadfolder, "{}_word2ind.pkl".format(self.language)))
         self.oov = np.max(list(self.word2ind.values()))

    def save_vocab(self, savefolder):
        os.makedirs(savefolder, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(savefolder, "{}_vectorizer.pkl".format(self.language)))
        joblib.dump(self.word2ind, os.path.join(savefolder, "{}_word2ind.pkl".format(self.language)))
