import argparse
import yaml
import warnings

import tensorflow as tf

from trainer import Trainer

with open('./conf/params.yaml', 'rb') as f:
    conf = yaml.safe_load(f)['model_params']

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, choices=['oracle', 'mf', 'rmf'])
parser.add_argument('--eps', default=5., type=float)
parser.add_argument('--pow_list', default=[1.], type=float, nargs='*')
parser.add_argument('--iters', default=5, type=int)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    eps = args.eps
    pow_list = args.pow_list
    dim = conf['dim']
    lam = conf['lam']
    eta = conf['eta']
    batch_size = conf['batch_size']
    max_iters = conf['max_iters']
    iters = args.iters
    model_name = args.model_name

    # run simulations
    trainer = Trainer(dim=dim, lam=lam, batch_size=batch_size,
                      max_iters=max_iters, eta=eta, model_name=model_name)
    trainer.run(iters=iters, eps=eps, pow_list=pow_list)

    print('\n', '=' * 25, '\n')
    print(f'Finished Running {model_name}!')
    print('\n', '=' * 25, '\n')
