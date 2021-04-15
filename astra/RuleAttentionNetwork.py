"""
Code for self-training with weak supervision.
Author: Giannis Karamanolakis (gkaraman@cs.columbia.edu)
"""

import math
import random
import numpy as np
from numpy.random import seed
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Embedding, Input, Dropout, Dense, Lambda


class RAN:
    """
    Rule Attention Network
      * Input: text embedding x, array of rule predictions
      * Output: aggregate label
    """

    def __init__(self, args, num_rules, logger=None, name='ran'):
        self.args = args
        self.name = name
        self.logger = logger
        self.manual_seed = args.seed
        tf.random.set_seed(self.manual_seed)
        self.datapath = args.datapath
        self.model_dir = args.logdir
        self.sup_batch_size = args.train_batch_size
        self.unsup_batch_size = args.unsup_batch_size
        self.sup_epochs = args.num_epochs
        self.unsup_epochs = args.num_unsup_epochs
        self.num_labels = args.num_labels
        self.num_rules = num_rules
        # max_rule_seq_length: used for efficiency (Note: no rules are discarded.)
        self.max_rule_seq_length = min(self.num_rules, args.max_rule_seq_length)

        # Using Student as an extra rule
        self.num_rules += 1
        self.max_rule_seq_length += 1
        self.student_rule_id = self.num_rules
        self.hard_student_rule = args.hard_student_rule
        self.preprocess = None
        self.trained = False
        self.xdim = -1
        self.ignore_student = False
        self.gpus = 1
        
    def init(self, rule_pred):
        # Initialize RAN as majority voting (all sources have equal weights)
        self.majority_model = MajorityVoter(num_labels=self.num_labels)
        return

    def postprocess_rule_preds(self, rule_pred, student_pred=None):
        """
        Converts rule predictions to appropriate format
        :param rule_pred: a 2D array of rule preds: num_examples x num_rules
        :return:
            rule_one_hot: a 2D mask matrix: 1 if rule applies otherwise 0
            rule_pred:    a 3D rule prediction matrix (N x num_rules x num_classes): converting class indices to one-hot vectors
                          # if a rule predicts -1, then pred = [0,...,0]
            student_pred: the soft predictions of a student network
        """
        max_rule_seq_length = self.max_rule_seq_length - 1  # -1: Using student as extra rule
        N = rule_pred.shape[0]
        rule_mask = (rule_pred != -1).astype(int)
        fired_rule_ids = [(np.nonzero(x)[0] + 1).tolist() for x in rule_mask]
        non_zero_rule_pred = []  
        for i, fired_rules in enumerate(fired_rule_ids):
            preds_i = [rule_pred[i, j-1] for j in fired_rules]
            preds_i = preds_i + [self.num_labels] * (max_rule_seq_length - len(preds_i))
            if len(preds_i) > max_rule_seq_length:
                self.logger.info("WARNING: Num firing rules = {} > max_rule_seq_length = {}".format(len(preds_i), max_rule_seq_length))
                preds_i = preds_i[:max_rule_seq_length]
            non_zero_rule_pred.append(preds_i)
        one_hot_rule_pred = tf.one_hot(non_zero_rule_pred, self.num_labels + 1).numpy()
        one_hot_rule_pred = one_hot_rule_pred[:, :, :-1]
        fired_rule_ids = [x + [0] * (max_rule_seq_length - len(x)) for x in fired_rule_ids]
        fired_rule_ids = np.array(fired_rule_ids)

        if student_pred is not None:
            mask_one = np.ones((N, 1))
            if student_pred.ndim > 2:
                student_pred = np.squeeze(student_pred, axis=None)
            if self.hard_student_rule:
                # Convert Student's soft probabilities to hard labels
                student_pred = to_one_hot(np.argmax(student_pred, axis=1), self.num_labels)
            student_pred = student_pred[..., np.newaxis, :]  # Add axis=1
            one_hot_rule_pred = np.concatenate([student_pred, one_hot_rule_pred], axis=1)
            rule_mask = np.concatenate([mask_one, rule_mask], axis=1)
            if not self.ignore_student:
                student_rule_id = np.ones((N, 1)) * self.student_rule_id
            else:
                student_rule_id = np.zeros((N, 1))
            fired_rule_ids = np.concatenate([student_rule_id, fired_rule_ids], axis=1)

        return rule_mask, fired_rule_ids, one_hot_rule_pred

    def train(self, x_train, rule_pred_train, y_train, x_dev=None, rule_pred_dev=None, y_dev=None,
              student_pred_train=None, student_pred_dev=None,
              x_unsup=None, rule_pred_unsup=None, student_pred_unsup=None):
        
        assert x_unsup is not None, "For SSL RAN you need to also provide unlabeled data... "
        
        if x_train is not None:
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            rule_one_hot_train, fired_rule_ids_train, rule_pred_train = self.postprocess_rule_preds(rule_pred_train, student_pred_train)
            self.logger.info("X Train Shape " + str(x_train.shape) + ' ' + str(rule_pred_train.shape) + ' ' + str(y_train.shape))
        else:
            rule_one_hot_train, fired_rule_ids_train, rule_pred_train = None, None, None

        if x_dev is not None:
            x_dev = np.array(x_dev)
            y_dev = np.array(y_dev)
            rule_one_hot_dev, fired_rule_ids_dev, rule_pred_dev = self.postprocess_rule_preds(rule_pred_dev, student_pred_dev)
            self.logger.info("X Dev Shape " + str(x_dev.shape) + ' ' + str(rule_pred_dev.shape) + ' ' + str(y_dev.shape))
        else:
            rule_one_hot_dev, fired_rule_ids_dev, rule_pred_dev = None, None, None

        x_unsup = np.array(x_unsup)
        rule_one_hot_unsup, fired_rule_ids_unsup, rule_pred_unsup = self.postprocess_rule_preds(rule_pred_unsup, student_pred_unsup)
        self.logger.info("X Unsup Shape " + str(x_unsup.shape) + ' ' + str(rule_pred_unsup.shape))


        if not self.trained or (x_train is not None and self.xdim != x_train.shape[1]):
            if self.trained and self.xdim != x_train.shape[1]:
                self.logger.info("WARNING: Changing dimensionality of x from {} to {}".format(self.xdim, x_train.shape[1]))
            self.xdim = x_train.shape[1] if x_train is not None else x_unsup.shape[1]
            self.model = construct_rule_network(self.xdim,
                                                num_rules=self.num_rules,
                                                num_labels=self.num_labels,
                                                max_rule_seq_length=self.max_rule_seq_length,
                                                seed=self.manual_seed)
        
        self.logger.info("\n\n\t\t*** Training RAN ***")
        loss_fn = MinEntropyLoss(batch_size=self.unsup_batch_size * self.gpus)  # SSLLoss()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=loss_fn,
                           metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
        
        self.model.fit(
            x=[x_unsup, fired_rule_ids_unsup, rule_pred_unsup],
            y=np.array([-1] * x_unsup.shape[0]),
            batch_size=self.unsup_batch_size * self.gpus,
            shuffle=True,
            epochs=self.sup_epochs,
            callbacks=[
                create_learning_rate_scheduler(max_learn_rate=1e-2, end_learn_rate=1e-5, warmup_epoch_count=20,
                                               total_epoch_count=self.sup_epochs),
                K.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
            ],
            validation_data=([x_dev, fired_rule_ids_dev, rule_pred_dev], y_dev))

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) 
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                             loss=loss_fn,  
                             metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

        self.model.fit(
            x=[x_train, fired_rule_ids_train, rule_pred_train], 
            y=y_train,
            batch_size=self.sup_batch_size * self.gpus,
            shuffle=True,
            epochs=self.sup_epochs,
            callbacks=[
                create_learning_rate_scheduler(max_learn_rate=1e-2, end_learn_rate=1e-5, warmup_epoch_count=20,
                                               total_epoch_count=self.sup_epochs),
                K.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
            ],
            validation_data=([x_dev, fired_rule_ids_dev, rule_pred_dev], y_dev))

        self.trained = True
        dev_loss = self.model.evaluate([x_dev, fired_rule_ids_dev, rule_pred_dev], y_dev)
        res = {}
        res['dev_loss'] = dev_loss
        return res

    def predict(self, rule_pred, student_features, student_pred=None):
        if not self.trained:
            return self.predict_majority(rule_pred)
        else:
            return self.predict_ran(student_features, rule_pred, student_pred=student_pred)

    def predict_majority(self, rule_pred):
        agg_labels = self.majority_model.predict(rule_pred)
        agg_proba = self.majority_model.predict_proba(rule_pred)
        return {
            'preds': agg_labels,
            'proba': agg_proba,
            "att_scores": None,
            "rule_mask": None
        }

    def predict_ran(self, x, rule_pred, student_pred=None, batch_size=128, prefix=""):
        x = np.array(x)
        if student_pred is None:
            random_pred = (rule_pred != -1).sum(axis=1) == 0
        else:
            random_pred = np.array([False] * rule_pred.shape[0])
        rule_mask, fired_rule_ids, rule_pred_one_hot = self.postprocess_rule_preds(rule_pred, student_pred)

        self.logger.info("RAN - Predicting labels for {} texts".format(x.shape[0]))
        y_pred = self.model.predict(
            [x, fired_rule_ids, rule_pred_one_hot],
            batch_size=batch_size
        )
        self.logger.info("DONE, Getting attention scores...".format(x.shape[0]))
        desiredOutputs = [self.model.get_layer("attention").output]
        newModel = tf.keras.Model(self.model.inputs, desiredOutputs)
        att_scores = newModel.predict(
            [x, fired_rule_ids, rule_pred_one_hot],
            batch_size=batch_size)

        preds = np.argmax(y_pred, axis=-1).flatten()
        dist_to_random = 0.0
        confidence_thres = 1 / self.num_labels + dist_to_random
        max_proba = np.max(y_pred, axis=-1).flatten()
        ignore_pred = max_proba < confidence_thres
        random_pred[ignore_pred] = True
        soft_proba = y_pred
        preds[random_pred] = -1
        return {
            'preds': preds,
            'proba': soft_proba,
            "att_scores": att_scores,
            "rule_mask": rule_mask,
        }

    def load(self, savefile):
        self.logger.info("loading rule attention network from {}".format(savefile))
        self.model.load_weights(savefile)

    def save(self, savefile):
        self.logger.info("Saving rule attention network at {}".format(savefile))
        self.model.save_weights(savefile)
        return


def construct_rule_network(student_emb_dim, num_rules, num_labels, dense_dropout=0.3, max_rule_seq_length=10, seed=42):
    # Rule Attention Network
    # encoder = TFBertModel.from_pretrained(model_type)
    student_embeddings = Input(shape=(student_emb_dim,), name="student_embeddings")
    rule_ids = Input(shape=(max_rule_seq_length,), dtype=tf.int32, name="rule_ids")
    rule_preds_onehot = Input(shape=(max_rule_seq_length, num_labels,), name="rule_preds")

    # x_hidden: batch_size x 128
    x_hidden = Dropout(dense_dropout)(student_embeddings)
    x_hidden = Dense(units=128, activation="relu", name="dense")(x_hidden)
    x_hidden = Dropout(dense_dropout)(x_hidden)

    # rule_embeddings_hidden: batch_size x 128 x max_rule_seq_length
    rule_embeddings = Embedding(num_rules+1, 128,
                                # embeddings_initializer='uniform',
                                embeddings_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                                embeddings_regularizer=None, activity_regularizer=None,
                                embeddings_constraint=None, mask_zero=True, input_length=max_rule_seq_length,
                                name="rule_embed")(rule_ids)
    # Rule bias parameters
    rule_biases = Embedding(num_rules+1, 1,
                            embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None,
                            embeddings_constraint=None, mask_zero=True, input_length=max_rule_seq_length,
                            name="rule_bias")(rule_ids)

    # Compute attention scores
    att_scores = tf.keras.layers.Dot(axes=[1, 2])([x_hidden, rule_embeddings])
    att_scores = tf.keras.layers.Add()([att_scores, tf.keras.backend.squeeze(rule_biases, axis=-1)])
    att_sigmoid_proba = Lambda(lambda x: tf.keras.activations.sigmoid(x), name='attention')(att_scores)
    outputs = tf.keras.layers.Dot(axes=[1, 1], name='raw_outputs')([att_sigmoid_proba, rule_preds_onehot])
    outputs = Lambda(lambda x: normalize_with_random_rule(x[0], x[1], x[2]), name='outputs_with_uniform')((outputs, att_sigmoid_proba, rule_preds_onehot))

    # Normalize Outputs
    outputs = Lambda(lambda x: l1_normalize(x, num_labels), name='normalized_outputs')(outputs)

    # Build Model
    model = tf.keras.Model(inputs=[student_embeddings, rule_ids, rule_preds_onehot], outputs=outputs)
    print(model.summary())
    return model


def MinEntropyLoss(batch_size):
    def loss(y_true, y_prob):
        per_example_loss = -y_prob * tf.math.log(y_prob)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)
    return loss


class MajorityVoter:
    """
    Predicts probabilities using the majority vote of the weak sources
    Code adapted from the Snorkel source:
    https://github.com/snorkel-team/snorkel/blob/b3b0669f716a7b3ed6cd573b57f3f8e12bcd495a/snorkel/labeling/model/baselines.py
    """
    def __init__(self, num_labels):
        self.num_labels = num_labels

    def predict(self, rule_pred):
        Y_probs = self.predict_proba(rule_pred)
        Y_p = self.probs_to_preds(Y_probs)
        return Y_p

    def predict_proba(self, rule_pred):
        n, m = rule_pred.shape
        pred = np.zeros((n, self.num_labels))
        for i in range(n):
            counts = np.zeros(self.num_labels)
            for j in range(m):
                if rule_pred[i, j] != -1:
                    counts[rule_pred[i, j]] += 1
            pred[i, :] = np.where(counts == max(counts), 1, 0)
        pred /= pred.sum(axis=1).reshape(-1, 1)
        return pred

    def probs_to_preds(self, probs):
        num_datapoints, num_classes = probs.shape
        Y_pred = np.empty(num_datapoints)
        diffs = np.abs(probs - probs.max(axis=1).reshape(-1, 1))

        for i in range(num_datapoints):
            max_idxs = np.where(diffs[i, :] < 1e-5)[0]
            if len(max_idxs) == 1:
                Y_pred[i] = max_idxs[0]
            else:
                Y_pred[i] = -1
        return Y_pred.astype(np.int)


def to_one_hot(x, num_classes):
    targets = np.array([x]).reshape(-1)
    return np.eye(num_classes)[targets]


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


def l1_normalize(x, num_labels):
    x = x + 1e-05  # avoid stability issues
    l1_norm = tf.keras.backend.stop_gradient(tf.keras.backend.sum(x, axis=-1))
    l1_norm = tf.keras.backend.repeat_elements(tf.keras.backend.expand_dims(l1_norm), num_labels, axis=-1)
    return x / l1_norm


def normalize_with_random_rule(output, att_sigmoid_proba, rule_preds_onehot):
    num_labels = rule_preds_onehot.shape[-1]
    sum_prob = tf.keras.backend.stop_gradient(tf.keras.backend.sum(rule_preds_onehot, axis=-1))
    rule_mask = tf.keras.backend.cast(sum_prob > 0, 'float32')
    num_rules = tf.keras.backend.cast(tf.keras.backend.sum(sum_prob, axis=-1), 'float32')
    masked_att_proba = att_sigmoid_proba * rule_mask
    sum_masked_att_proba = tf.keras.backend.sum(masked_att_proba, axis=-1)
    uniform_rule_att_proba = num_rules - sum_masked_att_proba
    uniform_vec = tf.ones((tf.shape(uniform_rule_att_proba)[0], num_labels)) / num_labels
    uniform_pred = tf.math.multiply(
        tf.keras.backend.repeat_elements(tf.keras.backend.expand_dims(uniform_rule_att_proba), num_labels, axis=-1),
        uniform_vec)
    output_with_uniform_rule = output+uniform_pred
    return output_with_uniform_rule