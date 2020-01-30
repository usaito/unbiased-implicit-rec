"""
Codes for visualizing results of the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""

import argparse
import sys
import warnings

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from visualizer import Visualizer

parser = argparse.ArgumentParser()

parser.add_argument('--eps', default=5., type=float,
                    help='epsilon for generating relevance parameter')
parser.add_argument('--pow_list', default=[1.], type=float, nargs='*',
                    help='pow_lister of theta for generating exposure parameter')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    # hyper-parameters
    eps = args.eps
    pow_list = args.pow_list

    # run simulations
    mlflow.set_experiment('semi-sys-wsdm')
    with mlflow.start_run() as run:
        visualizer = Visualizer()
        visualizer.plot_rel_pred_results(eps=eps, pow_list=pow_list)
        visualizer.plot_ranking_results(eps=eps, pow_list=pow_list)
        visualizer.plot_test_curves(eps=eps, pow_list=pow_list)

        mlflow.log_param('eps', eps)
        mlflow.log_param('pow_list', pow_list)

        mlflow.log_artifacts('../plots/results/')
        mlflow.log_artifacts('../plots/curves/test/')
