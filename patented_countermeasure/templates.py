import pickle as pic
from itertools import product
from typing import List
from time import time

from joblib import Parallel, delayed

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split, ShuffleSplit

from signal_strength import SignalStrength

from utils import *

class B4GaussianEstimator:

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.unique_y = np.unique(y)
        self.templates = np.zeros(len(self.unique_y), dtype=object)

        for lab in self.unique_y:
            data_lab = X[y == lab]
            
            mean = np.mean(data_lab, axis=0)
            cov  = np.nan_to_num(np.cov(data_lab, rowvar=False), nan=0.0)

            self.templates[np.where(self.unique_y == lab)[0][0]] = multivariate_normal(mean, cov, allow_singular=True)

        return self

    def predict(self, X: np.ndarray):
        return np.argmax(self.predict_proba(X), axis=1) # Maximum Likelihood
    
    def predict_proba(self, X: np.ndarray):
        return np.array([t.logpdf(X) for t in self.templates]).T # Likelihoods

def b4_train_test_split(*arrays, train_size: int, val_sizes: List[int]):
    return train_test_split(*arrays, train_size=train_size + max(val_sizes), random_state=0)

def b4_gridsearch_cv(signal_strengths: List[SignalStrength], n_folds: int, train_sizes: List[int], val_sizes: List[int], num_features: List[int], traces_total: np.ndarray, labels_total: np.ndarray, masking: bool=False):
    times = np.zeros((len(signal_strengths), len(train_sizes), len(val_sizes), len(num_features), n_folds), dtype=np.float64)
    results = np.zeros((len(signal_strengths), len(train_sizes), len(val_sizes), len(num_features), n_folds, 2), dtype=np.float64)
    
    for s, sig_strength in enumerate(signal_strengths):
        for ts, train_size in enumerate(train_sizes):
            traces_trainval, traces_test, labels_trainval, labels_test = b4_train_test_split(traces_total, labels_total, train_size=train_size, val_sizes=val_sizes)
            for m, val_size in enumerate(val_sizes):
                rs = ShuffleSplit(n_splits=n_folds, test_size=val_size, train_size=train_size, random_state=0)
                for cv, (train_index, val_index) in enumerate(rs.split(traces_trainval)):
                    print(f"CV {cv}", end="\r")
                    traces_train, traces_val = traces_trainval[train_index], traces_trainval[val_index]
                    labels_train, labels_val = labels_trainval[train_index], labels_trainval[val_index]
                    for n, n_feat in enumerate(num_features):
                        if not masking:
                            start = time()
                            sig_strength.fit(traces_train, labels_train if not masking else labels_train[:, 1], n_feat) # Masking: use first share for feature selection
                            features_train = sig_strength.transform(traces_train)
                            gaussian_est = B4GaussianEstimator().fit(features_train, labels_train if not masking else 16 * labels_train[:, 0] + labels_train[:, 1])
                            times[s, ts, m, n, cv] = time() - start

                            features_val = sig_strength.transform(traces_val)
                            predicted = gaussian_est.predict(features_val)
                            results[s, ts, m, n, cv] = np.count_nonzero(predicted == (labels_val if not masking else 16 * labels_val[:, 0] + labels_val[:, 1])) / labels_val.shape[0]
                        else:
                            start = time()
                            sig_strength.fit(traces_train, labels_train[:, 1], n_feat) # Masking: use first share for feature selection
                            features_train = sig_strength.transform(traces_train)
                            gaussian_est_0 = B4GaussianEstimator().fit(features_train, labels_train[:, 0])
                            gaussian_est_1 = B4GaussianEstimator().fit(features_train, labels_train[:, 1])
                            times[s, ts, m, n, cv] = time() - start

                            features_val = sig_strength.transform(traces_val)
                            predicted_0 = gaussian_est_0.predict(features_val)
                            results[s, ts, m, n, cv, 0] = np.count_nonzero(predicted_0 == labels_val[:, 0]) / labels_val.shape[0]
                            predicted_1 = gaussian_est_1.predict(features_val)
                            results[s, ts, m, n, cv, 1] = np.count_nonzero(predicted_1 == labels_val[:, 1]) / labels_val.shape[0]

                    del traces_train, traces_val, features_train, features_val
                for n, n_feat in enumerate(num_features):
                    print(f"{sig_strength} [num features: {n_feat}, train size: {train_size}, val size: {val_size}]: {np.mean(results[s, ts, m, n, :, 0]):#.4g} ± {np.std(results[s, ts, m, n, :, 0]):#.4g} ({results[s, ts, m, n, :, 0]}). Training in {np.mean(times[s, ts, m, n]):#.4g} ± {np.std(times[s, ts, m, n]):#.4g} seconds.")
                    print(f"{sig_strength} [num features: {n_feat}, train size: {train_size}, val size: {val_size}]: {np.mean(results[s, ts, m, n, :, 1]):#.4g} ± {np.std(results[s, ts, m, n, :, 1]):#.4g} ({results[s, ts, m, n, :, 1]}). Training in {np.mean(times[s, ts, m, n]):#.4g} ± {np.std(times[s, ts, m, n]):#.4g} seconds.")
            del traces_trainval, traces_test 
                    
    return times, results

def b4_train(sig_strength: SignalStrength, n_feat: int, traces_train: np.ndarray, labels_train: np.ndarray, masking: bool=False):
    sig_strength.fit(traces_train, labels_train if not masking else labels_train[:, 1], n_feat)
    features_train = sig_strength.transform(traces_train)
    if masking:
            assert labels_train.shape[1] == 2
            labels_train = 16 * labels_train[:, 0] + labels_train[:, 1]
    gaussian_est = B4GaussianEstimator().fit(features_train, labels_train)

    return gaussian_est
