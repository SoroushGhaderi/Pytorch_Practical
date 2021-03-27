import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

n_samples, n_features = X.shape
print(f"n_samples: {n_samples}, n_features: {n_features}")

