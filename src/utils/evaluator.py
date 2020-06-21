"""
Codes for evaluating implicit recommenders for the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
from typing import List

import numpy as np
import pandas as pd

from .metrics import average_precision_at_k, dcg_at_k, recall_at_k


class Model:
    """Load learned recommendation models."""

    def __init__(self, model_name: str):
        """Initialize Class."""
        self.user_embed = np.load(f'../logs/{model_name}/embeds/user_embed.npy')
        self.item_embed = np.load(f'../logs/{model_name}/embeds/item_embed.npy')
        self.user_bias = np.load(f'../logs/{model_name}/embeds/user_bias.npy')
        self.item_bias = np.load(f'../logs/{model_name}/embeds/item_bias.npy')

    def predict(self, users: np.array, items: np.array) -> np.array:
        """Predict scores for each user-item pairs."""
        # predict ranking score for each user
        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
        item_emb = self.item_embed[items]
        user_bias = self.user_bias[users]
        item_bias = self.item_bias[items]
        scores: np.ndarray = (user_emb @ item_emb.T).flatten() + user_bias + item_bias
        return scores


class Evaluator:
    """Evaluator."""

    def __init__(self, train: np.array, val: np.array, test: np.array, model_name: str) -> None:
        """Initialize class."""
        self.model = Model(model_name)
        self.model_name = model_name
        self.users = test[test[:, 4] == 1, 0]
        self.items = test[test[:, 4] == 1, 1]
        self.unique_items = np.unique(self.items)
        data = np.r_[train, val, test]
        self.pos_data = data[data[:, 4] == 1]

    def evaluate(self, eps: float, pow: float, num_negatives: int = 100,
                 k: List[int] = [i for i in np.arange(1, 11)]) -> None:
        """Evaluate a Recommender."""
        results = {}
        metrics = {'DCG': dcg_at_k, 'Recall': recall_at_k, 'MAP': average_precision_at_k}

        for _k in k:
            for metric in metrics:
                results[f'{metric}@{_k}'] = []

        np.random.seed(12345)
        for user in set(self.users):
            indices = self.users == user
            pos_items = self.items[indices]
            all_pos_items = self.pos_data[self.pos_data[:, 0] == user, 1]
            neg_items = np.random.permutation(np.setdiff1d(self.unique_items, all_pos_items))[:num_negatives]
            items = np.r_[pos_items, neg_items]
            ratings = np.r_[np.ones_like(pos_items), np.zeros_like(neg_items)]

            # predict ranking score for each user
            scores = self.model.predict(users=np.int(user), items=items.astype(np.int))
            for _k in k:
                for metric, metric_func in metrics.items():
                    results[f'{metric}@{_k}'].append(metric_func(ratings, scores, _k))

            self.results = pd.DataFrame(index=results.keys())
            self.results[f'{self.model.model_name}'] = list(map(np.mean, list(results.values())))
            self.results.to_csv(f'../logs/{self.model_name}/results/ranking_{eps}_{pow}.csv')
