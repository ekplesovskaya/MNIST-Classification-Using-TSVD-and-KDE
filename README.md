### Hierarchical Classification on the MNIST Dataset Using Truncated SVD and Kernel Density Estimation

The present paper introduces a novel approach based on a truncated SVD and kernel density estimation that outputs a comparable error rate on MNIST. In addition to that, a hierarchical classification framework is proposed, that allows to enhance the algorithm accuracy. The resulting algorithm outperforms the reproducible SVM accuracy, which is regarded as a state-of-the-art machine learning algorithm for MNIST. The key advantage of the proposed framework consists in a low computational cost: training on MNIST takes 5 seconds. Thus, the research results state that TSVD-KDE algorithm has the potential for being an efficient classification algorithm.

#### Comparison of different configurations of TSVD-KDE with other algorithms on the MNIST dataset

| Model | Error rate, % 
--- | --- 
TSVD-KDE with the default hyperparameters | 2.93
TSVD-KDE after hyperparameters tuning | 1.7
Hierarchical classification based on TSVD-KDE | **1.53**
Gaussian SVM with C = 5 and gamma = 0.05 | 1.63
Random Forest with the default hyperparameters | 2.96
XGBoost with the default hyperparameters | 2.2
CatBoost with the default hyperparameters | 2.6

#### Computational time for different algorithms on the MNIST dataset

| Model | Training time, sec | Prediction time, sec | Tuning time, hours
--- | --- | --- | ---
Hierarchical classification algorithm | **5** | 35 | ~ **77**
SVM | 712 | 159 | ~ 217
Random Forest | 29 | 0.32 | -
XGBoost | 170 | **0.04** | -
CatBoost | 467 | 1.41 | -

#### Research code description

MNIST dataset could be found at [`MNIST_dataset`](MNIST_dataset). Experiments with the different TSVD-KDE configurations could be run by [`tsvd_kde_experiment.py`](tsvd_kde_experiment.py). Experiments with the state-of-the-art classificators ae available at [`baseline_experimants.py`](baseline_experimants.py).


