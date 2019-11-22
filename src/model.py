"""
Implicit Recommender models used for the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf


class AbstractRecommender(metaclass=ABCMeta):
    """Abstract base class for evaluator class."""

    @abstractmethod
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        raise NotImplementedError()

    @abstractmethod
    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        raise NotImplementedError()

    @abstractmethod
    def create_losses(self) -> None:
        """Create the losses."""
        raise NotImplementedError()

    @abstractmethod
    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        raise NotImplementedError()


class MF(AbstractRecommender):
    """Matrix Factorization for generating semi-synthetic data."""

    def __init__(self, num_users: np.array, num_items: np.array, dim: int = 20, eta: float = 0.005) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.eta = eta

        # Build the graphs
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.labels = tf.placeholder(tf.float32, [None, 1], name='label_placeholder')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices
            self.user_embeddings = tf.get_variable(
                f'user_embeddings', shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings = tf.get_variable(
                f'item_embeddings', shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            # lookup embeddings
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(self.item_embeddings, self.items)

        with tf.variable_scope('prediction'):
            self.preds = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed), 1)
            self.preds = tf.expand_dims(self.preds, 1)

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            # naive mean-squared-loss and binary cross entropy loss.
            self.mse = tf.reduce_mean(
                tf.square(self.labels - self.preds))
            self.ce = - tf.reduce_mean(
                self.labels * tf.log(tf.sigmoid(self.preds))
                + (1 - self.labels) * tf.log(1. - self.preds))

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            # set Adam optimizers.
            self.apply_grads_mse = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.mse)
            self.apply_grads_ce = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.ce)


class ImplicitRecommender(AbstractRecommender):
    """Implicit Recommenders."""

    def __init__(self, num_users: np.array, num_items: np.array,
                 dim: int, lam: float, eta: float, weight: int = 1, clip: float = 0.0) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta
        self.clip = clip
        self.weight = weight

        # Build the graphs
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.scores = tf.placeholder(tf.float32, [None, 1], name='score_placeholder')
        self.labels = tf.placeholder(tf.float32, [None, 1], name='label_placeholder')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices and biases
            self.user_embeddings = tf.get_variable(
                'user_embeddings', shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            self.user_b = tf.Variable(
                tf.random_normal(shape=[self.num_users], stddev=0.01), name='user_b')
            self.item_embeddings = tf.get_variable(
                'item_embeddings', shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_b = tf.Variable(
                tf.random_normal(shape=[self.num_items], stddev=0.01), name='item_b')
            self.global_bias = tf.get_variable(
                'global_bias', [1], initializer=tf.constant_initializer(1e-3, dtype=tf.float32))

            # lookup embeddings and biases
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.u_bias = tf.nn.embedding_lookup(self.user_b, self.users)
            self.i_embed = tf.nn.embedding_lookup(self.item_embeddings, self.items)
            self.i_bias = tf.nn.embedding_lookup(self.item_b, self.items)

        with tf.variable_scope('prediction'):
            self.logits = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed), 1)
            self.logits = tf.add(self.logits, self.u_bias)
            self.logits = tf.add(self.logits, self.i_bias)
            self.logits = tf.add(self.logits, self.global_bias)
            self.preds = tf.sigmoid(tf.expand_dims(self.logits, 1), name='sigmoid_prediction')

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            # define the naive binary cross entropy loss.
            self.ce = - tf.reduce_mean(
                self.labels * tf.log(self.preds) + (1 - self.labels) * tf.log(1. - self.preds))
            # define the unbiased binary cross entropy loss in Eq. (9).
            scores = tf.clip_by_value(self.scores, clip_value_min=self.clip, clip_value_max=1.0)
            self.weighted_ce = - tf.reduce_sum(
                self.weight * (self.labels / scores) * tf.log(self.preds)
                + (1 - self.labels / scores) * tf.log(1. - self.preds)) / \
                tf.reduce_sum(self.weight * self.labels + (1 - self.labels))

            # add the L2-regularizer terms.
            reg_term_embeds = tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings)
            reg_term_biases = tf.nn.l2_loss(self.item_b) + tf.nn.l2_loss(self.user_b)
            self.loss = self.weighted_ce + self.lam * (reg_term_embeds + reg_term_biases)

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            # define GradientDescent Optimizer.
            self.apply_grads = tf.train.GradientDescentOptimizer(
                learning_rate=self.eta).minimize(self.loss)
