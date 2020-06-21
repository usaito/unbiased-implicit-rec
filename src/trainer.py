from pathlib import Path
from typing import List
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops

from utils.data_generator import generate_sys_data
from utils.evaluator import Evaluator
from utils.models import ImplicitRecommender


def rec_trainer(sess: tf.Session, model: ImplicitRecommender,
                train: np.ndarray, val: np.ndarray, test: np.ndarray,
                max_iters: int = 500, batch_size: int = 2**12, model_name: str = 'mf',
                eps: float = 5, pow: float = 1.0, num: int = 0) -> float:
    """Train and Evaluate Implicit Recommender."""
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # count the num of train-val data.
    num_train = train.shape[0]
    num_val = val.shape[0]
    # specify model type.
    oracle = 'oracle' in model_name
    ips = 'rmf' in model_name

    # train the given implicit recommender
    np.random.seed(12345)
    for _ in np.arange(max_iters):
        idx = np.random.choice(np.arange(num_train), size=batch_size)
        train_batch = train[idx]
        train_label = train_batch[:, 6] if oracle else train_batch[:, 2]
        val_label = val[:, 6] if oracle else val[:, 2]
        train_score = np.expand_dims(train_batch[:, 5], 1) if ips else np.ones((batch_size, 1))
        val_score = np.expand_dims(val[:, 5], 1) if ips else np.ones((num_val, 1))

        _, loss = sess.run([model.apply_grads, model.weighted_ce],
                           feed_dict={model.users: train_batch[:, 0], model.items: train_batch[:, 1],
                                      model.labels: np.expand_dims(train_label, 1), model.scores: train_score})
        val_loss = sess.run(model.weighted_ce, feed_dict={model.users: val[:, 0], model.items: val[:, 1],
                                                          model.labels: np.expand_dims(val_label, 1), model.scores: val_score})
        test_loss = sess.run(model.ce, feed_dict={model.users: test[:, 0], model.items: test[:, 1],
                                                  model.labels: np.expand_dims(test[:, 4], 1)})
        train_loss_list.append(loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)

    path = Path(f'../logs/{model_name}')
    # save embeddings.
    (path / 'embeds').mkdir(parents=True, exist_ok=True)
    u_emb, i_emb, u_bias, i_bias = sess.run([model.user_embeddings, model.item_embeddings, model.user_b, model.item_b])
    np.save(file=path / 'embeds/user_embed.npy', arr=u_emb)
    np.save(file=path / 'embeds/item_embed.npy', arr=i_emb)
    np.save(file=path / 'embeds/user_bias.npy', arr=u_bias)
    np.save(file=path / 'embeds/item_bias.npy', arr=i_bias)
    # save train and val loss curves.
    (path / 'loss').mkdir(parents=True, exist_ok=True)
    np.save(file=path / f'loss/train_{eps}_{pow}_{num}.npy', arr=train_loss_list)
    np.save(file=path / f'loss/val_{eps}_{pow}_{num}.npy', arr=val_loss_list)
    np.save(file=path / f'loss/test_{eps}_{pow}_{num}.npy', arr=test_loss_list)

    sess.close()

    return test_loss_list[np.argmin(val_loss_list)]


@dataclass
class Trainer:
    """Trainer Class for ImplicitRecommender."""
    dim: int = 5
    lam: float = 1e-5
    max_iters: int = 500
    batch_size: int = 12
    eta: float = 0.1
    model_name: str = 'oracle'

    def run(self, iters: int, eps: float, pow_list: List[float]) -> None:
        """Train implicit recommenders."""
        path = Path(f'../logs/{self.model_name}/results')
        path.mkdir(parents=True, exist_ok=True)

        results = []
        for pow in pow_list:
            # generate semi-synthetic data
            data = generate_sys_data(eps=eps, pow=pow)
            num_users = np.int(data[:, 0].max() + 1)
            num_items = np.int(data[:, 1].max() + 1)
            # data splitting
            train, test = data, data[data[:, 2] == 0, :]  # train-test split
            train, val = train_test_split(train, test_size=0.1, random_state=0)  # train-val split

            for i in np.arange(iters):
                # define the TF graph
                # different initialization of model parameters for each iteration
                tf.set_random_seed(i)
                ops.reset_default_graph()
                sess = tf.Session()
                # define the implicit recommender model
                rec = ImplicitRecommender(num_users=num_users, num_items=num_items,
                                          dim=self.dim, lam=self.lam, eta=self.eta)
                # train and evaluate the recommender
                score = rec_trainer(sess, model=rec, train=train, val=val, test=test,
                                    max_iters=self.max_iters, batch_size=2**self.batch_size,
                                    model_name=self.model_name, eps=eps, pow=pow, num=i)
                results.append(score)
                evaluator = Evaluator(train=train, val=val, test=test, model_name=self.model_name)
                evaluator.evaluate(eps=eps, pow=pow)
        np.save(path / f'eps_{eps}.npy', arr=np.array(results).reshape((len(pow_list), iters)).T)
