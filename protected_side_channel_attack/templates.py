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
    results = np.zeros((len(signal_strengths), len(train_sizes), len(val_sizes), len(num_features), n_folds), dtype=np.float64)
    
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
                        start = time()
                        sig_strength.fit(traces_train, labels_train if not masking else labels_train[:, 1], n_feat) # Masking: use first share for feature selection
                        features_train = sig_strength.transform(traces_train)
                        if masking:
                             assert labels_train.shape[1] == 2 and labels_val.shape[1] == 2
                             labels_train, labels_val = 16 * labels_train[:, 0] + labels_train[:, 1], 16 * labels_val[:, 0] + labels_val[:, 1]
                        gaussian_est = B4GaussianEstimator().fit(features_train, labels_train)
                        times[s, ts, m, n, cv] = time() - start

                        features_val = sig_strength.transform(traces_val)
                        predicted = gaussian_est.predict(features_val)
                        results[s, ts, m, n, cv] = np.count_nonzero(predicted == labels_val) / labels_val.shape[0]
                    del traces_train, traces_val, features_train, features_val
                for n, n_feat in enumerate(num_features):
                    print(f"{sig_strength} [num features: {n_feat}, train size: {train_size}, val size: {val_size}]: {np.mean(results[s, ts, m, n]):#.4g} ± {np.std(results[s, ts, m, n]):#.4g} ({results[s, ts, m, n]}). Training in {np.mean(times[s, ts, m, n]):#.4g} ± {np.std(times[s, ts, m, n]):#.4g} seconds.")
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

def get_masks_labels(seeds: np.ndarray, key: np.ndarray, key_shares: np.ndarray, round_perms: np.ndarray = None, copy_perms: np.ndarray = None):
    if round_perms is None or copy_perms is None:
        round_perms = np.zeros((seeds.shape[0]), dtype=int)
        copy_perms = np.zeros((seeds.shape[0], LATEST_ROUND - EARLIEST_ROUND), dtype=int)
    labels = np.zeros((LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4, NR_SHARES, seeds.shape[0]), dtype=int)

    for i, (seed, key_share, round_perm, copy_perm) in enumerate(zip(seeds, key_shares, round_perms, copy_perms)):
        indices, whitening = chacha_random_b4(seed)

        for round_idx, copy_p in zip(range(EARLIEST_ROUND, LATEST_ROUND), copy_perm):
            for block_idx in range(BLOCK_WIDTH_B4):
                permuted_keyround_index = ((round_perm + round_idx) % (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4)) * BLOCK_WIDTH_B4 + ((copy_p + block_idx) % BLOCK_WIDTH_B4)
                key_index = indices[permuted_keyround_index]

                assert np.sum(key_share[key_index]) % 16 == key[key_index]
                labels[round_idx, block_idx, :-1, i] = key_share[key_index, :-1]    
                labels[round_idx, block_idx, -1, i] = (key_share[key_index, -1] + whitening[permuted_keyround_index]) % 16

    return labels

def get_masks_labels_rws(seeds: np.ndarray, key: np.ndarray, key_shares: np.ndarray, rws_perms: np.ndarray = None):
    if rws_perms is None:
        rws_perms = np.zeros((seeds.shape[0]), dtype=int)
    labels = np.zeros((KEYROUND_WIDTH_B4, NR_SHARES, seeds.shape[0]), dtype=int)

    for i, (seed, key_share, rws_perm) in enumerate(zip(seeds, key_shares, rws_perms)):
        indices, whitening = chacha_random_b4(seed)

        for keyround_index in range(KEYROUND_WIDTH_B4):
            permuted_keyround_index = (rws_perm + keyround_index) % KEYROUND_WIDTH_B4
            key_index = indices[permuted_keyround_index]

            assert np.sum(key_share[key_index]) % 16 == key[key_index]
            labels[keyround_index, :-1, i] = key_share[key_index, :-1]
            labels[keyround_index, -1, i] = (key_share[key_index, -1] + whitening[permuted_keyround_index]) % 16

    return labels

def process_trace(seed, rws_perm_proba: np.ndarray, masks_rws_proba: np.ndarray, round_perm_proba: np.ndarray, copy_perm_proba: np.ndarray, masks_proba: np.ndarray, rws_keep_only: np.ndarray, masks_rws_keep_only: np.ndarray, round_keep_only: np.ndarray, copy_keep_only: np.ndarray, masks_keep_only: np.ndarray):
    assert round_perm_proba.shape[0] == KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 and copy_perm_proba.shape[1] == BLOCK_WIDTH_B4

    classifications_per_key_nibble = np.zeros((KEY_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.longfloat)
    if np.argmax(rws_perm_proba) not in rws_keep_only:
        return classifications_per_key_nibble
    if np.argmax(round_perm_proba) not in round_keep_only:
        return classifications_per_key_nibble

    indices, whitening = chacha_random_b4(seed)

    classifications_per_keyround_nibble = np.zeros((KEYROUND_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.longfloat)
    for target_keyround_index in range(KEYROUND_WIDTH_B4):
        target_round_idx = target_keyround_index // BLOCK_WIDTH_B4
        target_block_idx = target_keyround_index % BLOCK_WIDTH_B4

        for target_witness in KEY_ALPHABET:
            rws_classifications_hypotheses = np.full((KEYROUND_WIDTH_B4, len(KEY_ALPHABET) ** (NR_SHARES - 1)), -np.inf, dtype=np.longfloat)
            for keyround_idx in range(KEYROUND_WIDTH_B4):
                if np.argmax(masks_rws_proba[keyround_idx]) not in masks_rws_keep_only[keyround_idx]:
                    continue

                rws_perm = (target_keyround_index - keyround_idx) % KEYROUND_WIDTH_B4
                if rws_keep_only.shape[0] == rws_perm_proba.shape[0] and np.all(rws_keep_only != np.arange(rws_perm_proba.shape[0])) and np.argmax(rws_perm_proba) != rws_perm:
                    continue
                rws_proba = rws_perm_proba[rws_perm]
                for m, mask in enumerate(product(KEY_ALPHABET, repeat=NR_SHARES-1)):
                    masked_value = (target_witness + whitening[target_keyround_index] - np.sum(mask)) % 16
                    if masks_rws_keep_only[round_idx, block_idx].shape[0] == masks_rws_proba[round_idx, block_idx].shape[0] and np.all(masks_rws_keep_only[round_idx, block_idx] != np.arange(masks_rws_proba[round_idx, block_idx].shape[0])) and np.argmax(masks_rws_proba[round_idx, block_idx]) != len(KEY_ALPHABET) * m + masked_value:
                        continue
                    probas = masks_rws_proba[keyround_idx, len(KEY_ALPHABET) * m + masked_value]

                    rws_classifications_hypotheses[keyround_idx, m] = rws_proba + probas

            classifications_hypotheses = np.full((LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4, len(KEY_ALPHABET) ** (NR_SHARES - 1)), -np.inf, dtype=np.longfloat)
            for round_idx in range(EARLIEST_ROUND, LATEST_ROUND):
                if np.argmax(copy_perm_proba[round_idx]) not in copy_keep_only[round_idx]:
                    continue

                round_perm = (target_round_idx - round_idx) % (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4)
                if round_keep_only.shape[0] == round_perm_proba.shape[0] and np.all(round_keep_only != np.arange(round_perm_proba.shape[0])) and np.argmax(round_perm_proba) != round_perm:
                    continue
                round_proba = round_perm_proba[round_perm]
                for block_idx in range(BLOCK_WIDTH_B4):
                    if np.argmax(masks_proba[round_idx, block_idx]) not in masks_keep_only[round_idx, block_idx]:
                        continue

                    copy_perm = (target_block_idx - block_idx) % BLOCK_WIDTH_B4
                    if copy_keep_only[round_idx].shape[0] == copy_perm_proba[round_idx].shape[0] and np.all(copy_keep_only[round_idx] != np.arange(copy_perm_proba[round_idx].shape[0])) and np.argmax(copy_perm_proba[round_idx]) != copy_perm:
                        continue
                    copy_proba = copy_perm_proba[round_idx, copy_perm]
                    for m, mask in enumerate(product(KEY_ALPHABET, repeat=NR_SHARES-1)):
                        masked_value = (target_witness + whitening[target_keyround_index] - np.sum(mask)) % 16
                        if masks_keep_only[round_idx, block_idx].shape[0] == masks_proba[round_idx, block_idx].shape[0] and np.all(masks_keep_only[round_idx, block_idx] != np.arange(masks_proba[round_idx, block_idx].shape[0])) and np.argmax(masks_proba[round_idx, block_idx]) != len(KEY_ALPHABET) * m + masked_value:
                            continue
                        probas = masks_proba[round_idx, block_idx, len(KEY_ALPHABET) * m + masked_value]

                        classifications_hypotheses[round_idx, block_idx, m] = round_proba + copy_proba + probas

            classifications_per_keyround_nibble[target_keyround_index, target_witness] = np.nan_to_num(np.logaddexp.reduce(rws_classifications_hypotheses, axis=None), neginf=0) + np.nan_to_num(np.logaddexp.reduce(classifications_hypotheses, axis=None), neginf=0)

    classifications_per_key_nibble[indices[:KEYROUND_WIDTH_B4]] = classifications_per_keyround_nibble
    return classifications_per_key_nibble

def classifications_per_trace(seeds: np.ndarray, rws_perm_probas: np.ndarray, masks_rws_probas: np.ndarray, round_perm_probas: np.ndarray, copy_perm_probas: np.ndarray, masks_probas: np.ndarray, rws_keep_only: np.ndarray = None, masks_rws_keep_only: np.ndarray = None, round_keep_only: np.ndarray = None, copy_keep_only: np.ndarray = None, masks_keep_only: np.ndarray = None, parallel: bool = True):
    if rws_keep_only is None:
        rws_keep_only = np.arange(rws_perm_probas.shape[1])
    if masks_rws_keep_only is None:
        masks_rws_keep_only = np.tile(np.arange(masks_rws_probas.shape[2]), masks_rws_probas.shape[1:-1] + (1,))
    if round_keep_only is None:
        round_keep_only = np.arange(round_perm_probas.shape[1])
    if copy_keep_only is None:
        copy_keep_only = np.tile(np.arange(copy_perm_probas.shape[2]), copy_perm_probas.shape[1:-1] + (1,))
    if masks_keep_only is None:
        masks_keep_only = np.tile(np.arange(masks_probas.shape[3]), masks_probas.shape[1:-1] + (1,))

    if parallel:
        per_trace = np.array(Parallel(n_jobs=-1)(delayed(process_trace)(seed, rws_perm_proba, masks_rws_proba, round_perm_proba, copy_perm_proba, masks_proba, rws_keep_only, masks_rws_keep_only, round_keep_only, copy_keep_only, masks_keep_only) for seed, rws_perm_proba, masks_rws_proba, round_perm_proba, copy_perm_proba, masks_proba in zip(seeds, rws_perm_probas, masks_rws_probas, round_perm_probas, copy_perm_probas, masks_probas)))
    else:
        LIMIT = 5
        per_trace = np.array([process_trace(seed, rws_perm_proba, masks_rws_proba, round_perm_proba, copy_perm_proba, masks_proba, rws_keep_only, masks_rws_keep_only, round_keep_only, copy_keep_only, masks_keep_only) for seed, rws_perm_proba, masks_rws_proba, round_perm_proba, copy_perm_proba, masks_proba in zip(seeds[:LIMIT], rws_perm_probas[:LIMIT], masks_rws_probas[:LIMIT], round_perm_probas[:LIMIT], copy_perm_probas[:LIMIT], masks_probas[:LIMIT])])

    with open(per_trace_filepath, "wb") as w:
        pic.dump(per_trace, w)

    return per_trace

def reconstruct_key(per_trace: np.ndarray):
    classifications_per_key_nibble = np.sum(per_trace, axis=0)
    recovered_key = np.argmax(classifications_per_key_nibble, axis=1)

    return recovered_key

def guessing_entropy(classifications_per_key_nibble: np.ndarray, key: np.ndarray):
    return np.argmax(np.argsort(classifications_per_key_nibble, axis=-1)[..., ::-1] == key[..., np.newaxis], axis=-1)
