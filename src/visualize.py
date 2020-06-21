import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from plotly.graph_objs import Figure, Layout, Scatter
from plotly.offline import plot

metrics = ['DCG', 'Recall', 'MAP']
models = ['oracle', 'mf', 'rmf']
names = ['Oracle', 'Naive', 'Unbiased']
colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
          'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
          'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
          'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
          'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

fill_colors = ['rgba(31, 119, 180, 0.2)', 'rgba(255, 127, 14, 0.2)',
               'rgba(44, 160, 44, 0.2)', 'rgba(214, 39, 40, 0.2)',
               'rgba(148, 103, 189, 0.2)', 'rgba(140, 86, 75, 0.2)',
               'rgba(227, 119, 194, 0.2)', 'rgba(127, 127, 127, 0.2)',
               'rgba(188, 189, 34, 0.2)', 'rgba(23, 190, 207, 0.2)']


@dataclass
class Visualizer:
    """Class for visualizing experimental results."""
    iters: int = 5

    def plot_rel_pred_results(self, eps: float, pow_list: List[float]) -> None:
        """Summarize results of the experiments."""
        mean_results = np.concatenate([
            (np.load(f'../logs/{model}/results/eps_{eps}.npy') / np.load(f'../logs/oracle/results/eps_{eps}.npy')).mean(0)
            for model in models]).reshape((len(models), self.iters))
        se_results = np.concatenate([
            (np.load(f'../logs/{model}/results/eps_{eps}.npy') / np.load(f'../logs/oracle/results/eps_{eps}.npy')).std(0)
            for model in models]).reshape((len(models), self.iters))
        upper = mean_results + 2 * se_results / np.sqrt(self.iters)
        lower = mean_results - 2 * se_results / np.sqrt(self.iters)

        scat_list = []
        for i, model in enumerate(models):
            scat_list.append(Scatter(x=pow_list, y=mean_results[i, :], name=names[i],
                                     mode='lines+markers', marker=dict(size=15),
                                     line=dict(color=colors[i], width=6)))
        for i, model in enumerate(models):
            scat_list.append(Scatter(x=pow_list + pow_list[::-1],
                                     y=np.r_[upper[i], lower[i][::-1]], name=names[i],
                                     fill='tozerox', fillcolor=fill_colors[i],
                                     line=dict(color='rgba(255,255,255,0)'), showlegend=False))

        layout_rel_pred.title.text = f'Epsilon = {eps}'
        Path('../plots/results/rel_pred').mkdir(parents=True, exist_ok=True)
        plot(Figure(data=scat_list, layout=layout_rel_pred), auto_open=False,
             filename=f'../plots/results/rel_pred/{eps}.html')

    def plot_ranking_results(self, eps: float, pow_list: List[float]) -> None:
        """Plot results with varying K."""
        self._load_and_save_results(models=models, eps=eps, pow_list=pow_list)
        for pow in pow_list:
            for met in metrics:
                ret = pd.read_csv(f'../logs/overall/results/ranking_{eps}_{pow}.csv',
                                  index_col=0).T[[f'{met}@{k}' for k in [i for i in np.arange(1, 11)]]]
                scatter_list = [Scatter(x=[i for i in np.arange(1, 11)], y=np.array(ret)[i, :],
                                        marker=dict(size=20), line=dict(width=6), name=names[i])
                                for i in np.arange(len(models))]

                layout_overall_results.title.text = f'{met}: Power of Theta (p) = {pow}'
                Path('../plots/results/ranking').mkdir(parents=True, exist_ok=True)
                plot(Figure(data=scatter_list, layout=layout_overall_results), auto_open=False,
                     filename=f'../plots/results/ranking/{met}_{eps}_{pow}_K.html')

    def plot_test_curves(self, eps: float, pow_list: List[float]) -> None:
        """Plot test curves."""
        for pow in pow_list:
            scatter_list = []
            for i, model in enumerate(models):
                test = np.concatenate(
                    [np.expand_dims(np.load(f'../logs/{model}/loss/test_{eps}_{pow}_{i}.npy'), 1)
                     for i in np.arange(self.iters)], 1)

                scatter_list.append(Scatter(x=np.arange(len(test)), y=test.mean(1), name=names[i], opacity=0.8,
                                            mode='lines', line=dict(color=colors[i], width=6)))
                scatter_list.append(Scatter(x=np.r_[np.arange(len(test)), np.arange(len(test))[::-1]],
                                            y=np.r_[test.mean(1) + test.std(1) / np.sqrt(self.iters),
                                                    (test.mean(1) - test.std(1) / np.sqrt(self.iters))[::-1]],
                                            mode='lines', opacity=0.8, name=names[i],
                                            fill='tozerox', fillcolor=fill_colors[i],
                                            line=dict(color='rgba(255,255,255,0)'), showlegend=False))

            layout_test_curves.title.text = f'Power of Theta (p) = {pow}'
            Path('../plots/curves/test').mkdir(parents=True, exist_ok=True)
            plot(Figure(data=scatter_list, layout=layout_test_curves), auto_open=False,
                 filename=f'../plots/curves/test/test_curves_{eps}_{pow}.html')

    def _load_and_save_results(self, models: List[str], eps: float, pow_list: List[float]) -> None:
        """Load and save experimental results."""
        path = Path('../logs/overall/results')
        path.mkdir(parents=True, exist_ok=True)

        for pow in pow_list:
            aoa_list = []
            for model in models:
                aoa_list.append(pd.read_csv(f'../logs/{model}/results/ranking_{eps}_{pow}.csv', index_col=0))
            pd.concat([aoa_list[i] for i in np.arange(len(models))], 1).to_csv(path / f'ranking_{eps}_{pow}.csv')


# define layouts
layout_rel_pred = Layout(
    title=dict(font=dict(size=40), x=0.5, xanchor='center', y=0.99),
    paper_bgcolor='rgb(255,255,255)', plot_bgcolor='rgb(235,235,235)', width=1000, height=800,
    xaxis=dict(title='Power of Theta (p)', range=[0.49, 4.01], titlefont=dict(size=30),
               tickfont=dict(size=18), dtick=0.5, gridcolor='rgb(255,255,255)'),
    yaxis=dict(title='relative log loss', range=[0.95, 1.6], titlefont=dict(
        size=30), tickfont=dict(size=18), gridcolor='rgb(255,255,255)'),
    legend=dict(bgcolor='rgb(245,245,245)', x=0.99, xanchor='right',
                        orientation='h', y=0.99, yanchor='top', font=dict(size=30)),
    margin=dict(l=80, t=50, b=60))


layout_overall_results = Layout(
    title=dict(font=dict(size=40), x=0.5, xanchor='center', y=0.99),
    paper_bgcolor='rgb(255,255,255)', plot_bgcolor='rgb(235,235,235)', width=1000, height=800,
    xaxis=dict(title='varying K', titlefont=dict(size=30),
               tickmode='array', tickvals=[i for i in np.arange(1, 11)],
               ticktext=[i for i in np.arange(1, 11)],
               tickfont=dict(size=20), gridcolor='rgb(255,255,255)'),
    yaxis=dict(tickfont=dict(size=15), gridcolor='rgb(255,255,255)'),
    legend=dict(bgcolor='rgb(245,245,245)', x=0.01, xanchor='left',
                orientation='h', y=0.99, yanchor='top', font=dict(size=32)),
    margin=dict(l=50, t=50, b=60))

layout_test_curves = Layout(
    title=dict(font=dict(size=40), x=0.5, xanchor='center', y=0.99),
    paper_bgcolor='rgb(255,255,255)', plot_bgcolor='rgb(235,235,235)', width=1200, height=800,
    xaxis=dict(title='iterations', titlefont=dict(size=30),
               tickfont=dict(size=18), gridcolor='rgb(255,255,255)'),
    yaxis=dict(title='log loss', titlefont=dict(size=30),
               tickfont=dict(size=18), gridcolor='rgb(255,255,255)'),
    legend=dict(bgcolor='rgb(245,245,245)', x=0.99, xanchor='right',
                orientation='h', y=0.99, yanchor='top', font=dict(size=30)),
    margin=dict(l=80, t=50, b=60))


parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=float)
parser.add_argument('--pow_list', type=float, nargs='*')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    eps = args.eps
    pow_list = args.pow_list

    visualizer = Visualizer()
    visualizer.plot_rel_pred_results(eps=eps, pow_list=pow_list)
    visualizer.plot_ranking_results(eps=eps, pow_list=pow_list)
    visualizer.plot_test_curves(eps=eps, pow_list=pow_list)
