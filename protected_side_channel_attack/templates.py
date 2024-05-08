from typing import List
from time import time

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split, ShuffleSplit

from signal_strength import SignalStrength

class B4GaussianEstimator:

    def fit(self, X: np.ndarray, y: np.ndarray):
        unique_y = np.unique(y)
        assert set(unique_y) == set(range(len(unique_y)))
        self.templates = np.zeros(len(unique_y), dtype=object)

        for lab in unique_y:
            data_lab = X[y == lab]
            
            mean = np.mean(data_lab, axis=0)
            cov  = np.cov(data_lab, rowvar=False)

            self.templates[lab] = multivariate_normal(mean, cov)

        return self

    def predict(self, X: np.ndarray):
        return np.argmax(self.predict_proba(X))
    
    def predict_proba(self, X: np.ndarray):
        return np.array([t.logpdf(X) for t in self.templates])

def b4_gridsearch_cv(signal_strengths: List[SignalStrength], n_folds: int, train_sizes: List[int], val_sizes: List[int], num_features: List[int], traces_total: np.ndarray, labels_total: np.ndarray):
    times = np.zeros((len(signal_strengths), len(train_sizes), len(val_sizes), len(num_features), n_folds), dtype=np.float64)
    results = np.zeros((len(signal_strengths), len(train_sizes), len(val_sizes), len(num_features), n_folds), dtype=np.float64)
    
    for s, sig_strength in enumerate(signal_strengths):
        for ts, train_size in enumerate(train_sizes):
            traces_trainval, traces_test, labels_trainval, labels_test = train_test_split(traces_total, labels_total, train_size=train_size + max(val_sizes), random_state=0)
            for m, val_size in enumerate(val_sizes):
                rs = ShuffleSplit(n_splits=n_folds, test_size=val_size, train_size=train_size, random_state=0)
                for n, n_feat in enumerate(num_features):
                    for cv, (train_index, val_index) in enumerate(rs.split(traces_trainval)):
                        print(f"CV {cv}", end="\r")
                        traces_train, traces_val = traces_trainval[train_index], traces_trainval[val_index]
                        labels_train, labels_val = labels_trainval[train_index], labels_trainval[val_index]

                        start = time()
                        sig_strength.fit(traces_train, labels_train, n_feat)
                        features_train = sig_strength.transform(traces_train)
                        gaussian_est = B4GaussianEstimator().fit(features_train, labels_train)
                        times[s, ts, m, n, cv] = time() - start

                        features_val = sig_strength.transform(traces_val)
                        predicted = gaussian_est.predict(features_val)
                        results[s, ts, m, n, cv] = np.count_nonzero(predicted == labels_val) / labels_val.shape[0]
                        del traces_train, traces_val, features_train, features_val

                    print(f"{sig_strength} [num features: {n_feat}, train size: {train_size}, val size: {val_size}]: {np.mean(results[s, ts, m, n]):#.4g} ± {np.std(results[s, ts, m, n]):#.4g} ({results[s, ts, m, n]}). Training in {np.mean(times[s, ts, m, n]):#.4g} ± {np.std(times[s, ts, m, n]):#.4g} seconds.")
            del traces_trainval, traces_test 
                    
    return times, results