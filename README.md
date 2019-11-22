## Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback
---

### About

This repository accompanies the semi-synthetic simulation conducted in the paper "[Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback](https://arxiv.org/abs/1909.03601)" by Yuta Saito, Suguru Yaginuma, Yuta Nishino, Hayato Sakata, and Kazuhide Nakata, which has been accepted to [WSDM'20](http://www.wsdm-conference.org/2020/index.php).

If you find this code useful in your research then please cite:
```
@inproceedings{saito2020unbiased,
  title={Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback},
  author={Yuta Saito, Suguru Yaginuma, Yuta Nishino, Hayato Sakata, and Kazuhide Nakata},
  booktitle={Proceedings of the Thirteenth ACM International Conference on Web Search and Data Mining},
  year={2020},
  organization={ACM}
}
```

### Dependencies
  - python==3.7.3
  - numpy==1.16.2
  - pandas==0.24.2
  - scikit-learn==0.20.3
  - tensorflow==1.14.0
  - plotly==3.10.0   
  - mlflow==1.3.0

### Running the code
To run the simulation with semi-synthetic data, download MovieLens 100K dataset from (https://grouplens.org/datasets/movielens/) and rename the `u.data` as `ml-100k.data` and put it into `data/ml-100k` directory. Then, navigate to the `src/` directory and run the command

```
$ sh run.sh
```

This will run semi-synthetic experiments conducted in Section 5 with a fixed value of epsilon (=5) over 5 different values of power of theta (0.5, 1, 2, 3, 4). Besides, Figure1 will be generated.

You can see the default values in the `run.sh` file; those values are actually used in our experiments.

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
$ mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

### Figures

By running the codes above, you can obtain the figures below.

| Figure 1: Epsilon=0.5 | Figure 1: Power of Theta=0.5 | Figure 1: Power of Theta=2.0 | Figure 1: Power of Theta=4.0 |
|:-: | :-: |:-: | :-: |
|<img src="./image/eps-5.png"> | <img src="./image/dcg-05.png">| <img src="./image/dcg-2.png">| <img src="./image/dcg-4.png">|

You will also have the results of other ranking metrics including MAP and Recall.
