"""Evaluation metrics for Collaborative Filltering with Implicit Feedback."""
import numpy as np


def dcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Calculate a DCG score."""
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]

    dcg_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        dcg_score += y_true_sorted_by_score[0]
        for i in np.arange(1, k):
            dcg_score += y_true_sorted_by_score[i] / np.log2(i + 1)

    return dcg_score / np.sum(y_true_sorted_by_score)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Calculate a precision score."""
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]][:k]

    precision_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        precision_score = np.mean(y_true_sorted_by_score)

    return precision_score


def average_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Calculate a average precision for a user."""
    y_true_sorted_by_score: np.ndarray = y_true[y_score.argsort()[::-1]]

    average_precision_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        for i in np.arange(k):
            if y_true_sorted_by_score[i] == 1:
                average_precision_score += np.sum(y_true_sorted_by_score[:i + 1]) / (i + 1)

    return average_precision_score / np.sum(y_true_sorted_by_score)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Calculate a recall score."""
    y_true_sorted_by_score: np.ndarray = y_true[y_score.argsort()[::-1]]

    recall_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        recall_score = np.sum(y_true_sorted_by_score[:k]) / np.sum(y_true_sorted_by_score)

    return recall_score
