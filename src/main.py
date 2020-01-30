"""
Codes for running the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""

import argparse
import sys
import warnings

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from trainer import Trainer

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str,
                    choices=['oracle', 'mf', 'rmf'], help='a used model')
parser.add_argument('--eps', default=5., type=float,
                    help='epsilon for generating relevance parameter')
parser.add_argument('--pow_list', default=[1.], type=float, nargs='*',
                    help='pow_lister of theta for generating exposure parameter')
parser.add_argument('--dim', default=10, type=int,
                    help='dim of user-item latent factors')
parser.add_argument('--lam', default=1e-5, type=float,
                    help='weight of l2 reguralization')
parser.add_argument('--eta', default=1e-1, type=float,
                    help='learning_rate for SGD')
parser.add_argument('--batch_size', default=12, type=int,
                    help='batch_size for mini-batch sampling')
parser.add_argument('--max_iters', default=500, type=int,
                    help='maximun num of iterations for SGD')
parser.add_argument('--iters', default=5, type=int, help='num of simulations')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    # hyper-parameters
    eps = args.eps
    pow_list = args.pow_list
    dim = args.dim
    lam = args.lam
    eta = args.eta
    batch_size = args.batch_size
    max_iters = args.max_iters
    iters = args.iters
    model_name = args.model_name

    # run simulations
    mlflow.set_experiment('semi-sys-wsdm')
    with mlflow.start_run() as run:
        trainer = Trainer(
            dim=dim, lam=lam, batch_size=batch_size,
            max_iters=max_iters, eta=eta, model_name=model_name)
        trainer.run(eps=eps, pow_list=pow_list)

        print('\n', '=' * 25, '\n')
        print(f'Finished Running {model_name}!')
        print('\n', '=' * 25, '\n')

        mlflow.log_param('eps', eps)
        mlflow.log_param('pow_list', pow_list)
        mlflow.log_param('dim', dim)
        mlflow.log_param('lam', lam)
        mlflow.log_param('eta', eta)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('max_iters', max_iters)
        mlflow.log_param('iters', iters)
        mlflow.log_param('model_name', model_name)

        mlflow.log_artifacts(f'../logs/{model_name}/results/')
