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

def get_rws_masks_and_round_masks(seeds: np.ndarray, key: np.ndarray, key_shares: np.ndarray, rws_perms: np.ndarray = None, round_perms: np.ndarray = None, copy_perms: np.ndarray = None):
    if rws_perms is None or round_perms is None or copy_perms is None:
        rws_perms = np.zeros((seeds.shape[0]), dtype=int)
        round_perms = np.zeros((seeds.shape[0]), dtype=int)
        copy_perms = np.zeros((seeds.shape[0], LATEST_ROUND - EARLIEST_ROUND), dtype=int)
    rws_masks = np.zeros((KEYROUND_WIDTH_B4, NR_SHARES, seeds.shape[0]), dtype=int)
    round_masks = np.zeros((LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4, NR_SHARES, seeds.shape[0]), dtype=int)

    for i, (seed, key_share, rws_perm, round_perm, copy_perm) in enumerate(zip(seeds, key_shares, rws_perms, round_perms, copy_perms)):
        indices, whitening = chacha_random_b4(seed)
        for keyround_index in range(KEYROUND_WIDTH_B4):
            permuted_keyround_index = (rws_perm + keyround_index) % KEYROUND_WIDTH_B4
            key_index = indices[permuted_keyround_index]

            assert np.sum(key_share[key_index]) % 16 == key[key_index]
            rws_masks[keyround_index, :-1, i] = key_share[key_index, :-1]
            rws_masks[keyround_index, -1, i] = (key_share[key_index, -1] + whitening[permuted_keyround_index]) % 16

            round_idx = keyround_index // BLOCK_WIDTH_B4
            block_idx = keyround_index % BLOCK_WIDTH_B4
            if not (EARLIEST_ROUND <= round_idx < LATEST_ROUND):
                continue
            copy_p = copy_perm[round_idx]
            permuted_keyround_index = ((round_perm + round_idx) % (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4)) * BLOCK_WIDTH_B4 + ((copy_p + block_idx) % BLOCK_WIDTH_B4)
            key_index = indices[permuted_keyround_index]

            assert np.sum(key_share[key_index]) % 16 == key[key_index]
            round_masks[round_idx, block_idx, :-1, i] = key_share[key_index, :-1]    
            round_masks[round_idx, block_idx, -1, i] = (key_share[key_index, -1] + whitening[permuted_keyround_index]) % 16
    return rws_masks, round_masks

def get_all_labels(seeds: np.ndarray, key: np.ndarray, log_output: np.ndarray):
    keyshares, perms = log_output
    rws_perms_labels = perms[:, 0]
    round_perms_labels = perms[:, 1]
    copy_perms_labels = perms[:, 2::8].T
    rws_masks_labels, round_masks_labels = get_rws_masks_and_round_masks(seeds, key, keyshares, rws_perms_labels, round_perms_labels, copy_perms_labels.T)
    
    return rws_perms_labels, rws_masks_labels, round_perms_labels, copy_perms_labels, round_masks_labels

def process_trace(seed, round_perm_proba: np.ndarray, copy_perm_proba: np.ndarray, masks_proba: np.ndarray):
    assert round_perm_proba.shape == (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4,) and copy_perm_proba.shape == (LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4) and (masks_proba.shape == (LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4, len(KEY_ALPHABET) ** NR_SHARES) or masks_proba.shape == (LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4, NR_SHARES, len(KEY_ALPHABET)))
    predicted_key = np.zeros((KEY_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.float64)

    if masks_proba.ndim == 3:
        masks_proba = np.concatenate((np.full((EARLIEST_ROUND, BLOCK_WIDTH_B4, len(KEY_ALPHABET) ** NR_SHARES), np.log(1/(len(KEY_ALPHABET) ** NR_SHARES))), masks_proba, np.full((KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 - LATEST_ROUND, BLOCK_WIDTH_B4, len(KEY_ALPHABET) ** NR_SHARES), np.log(1/(len(KEY_ALPHABET) ** NR_SHARES)))), axis=0)
        predicted_values = np.stack([np.logaddexp.reduce([masks_proba[:, :, len(KEY_ALPHABET) * m + ((k - m) % 16)] for m in KEY_ALPHABET], axis=0) for k in KEY_ALPHABET], axis=-1)
    elif masks_proba.ndim == 4:
        masks_proba = np.concatenate((np.full((EARLIEST_ROUND, BLOCK_WIDTH_B4, NR_SHARES, len(KEY_ALPHABET)), np.log(1/len(KEY_ALPHABET))), masks_proba, np.full((KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 - LATEST_ROUND, BLOCK_WIDTH_B4, NR_SHARES, len(KEY_ALPHABET)), np.log(1/len(KEY_ALPHABET)))), axis=0)
        predicted_values = np.stack([np.logaddexp.reduce([masks_proba[:, :, 0, m] + masks_proba[:, :, 1, ((k - m) % 16)] for m in KEY_ALPHABET], axis=0) for k in KEY_ALPHABET], axis=-1)
    else:
        raise ValueError

    copy_perm_proba = np.concatenate((np.full((EARLIEST_ROUND, BLOCK_WIDTH_B4), np.log(1/BLOCK_WIDTH_B4)), copy_perm_proba, np.full((KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 - LATEST_ROUND, BLOCK_WIDTH_B4), np.log(1/BLOCK_WIDTH_B4))), axis=0)
    predicted_rounds = np.zeros((KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4, BLOCK_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.float64)
    for round_idx in range(KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4):
        for block_idx in range(BLOCK_WIDTH_B4):
            for k in KEY_ALPHABET:
                predicted_rounds[round_idx, block_idx, k] = np.logaddexp.reduce([predicted_values[round_idx, (block_idx - copy_perm) % BLOCK_WIDTH_B4, k] + copy_perm_proba[round_idx, copy_perm] for copy_perm in range(BLOCK_WIDTH_B4)], axis=0)

    predicted_total = np.zeros((KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4, BLOCK_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.float64)
    for round_idx in range(KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4):
        for block_idx in range(BLOCK_WIDTH_B4):
            for k in KEY_ALPHABET:
                predicted_total[round_idx, block_idx, k] = np.logaddexp.reduce([predicted_rounds[(round_idx - round_perm) % (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4), block_idx, k] + round_perm_proba[round_perm] for round_perm in range(KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4)], axis=0)

    predicted_keyround = np.concatenate(predicted_total, axis=0) # (KEYROUND_WIDTH_B4, KEY_ALPHABET)
    indices, whitening = chacha_random_b4(seed)
    predicted_key[indices[:KEYROUND_WIDTH_B4]] = np.array([np.roll(pred, -whi) for pred, whi in zip(predicted_keyround, whitening)])

    return predicted_key

def classifications_per_trace(per_trace_filepath: str, seeds: np.ndarray, round_perm_probas: np.ndarray, copy_perm_probas: np.ndarray, masks_probas: np.ndarray, parallel: bool = True):
    if parallel:
        per_trace = np.array(Parallel(n_jobs=-1)(delayed(process_trace)(seed, round_perm_proba, copy_perm_proba, masks_proba) for seed, round_perm_proba, copy_perm_proba, masks_proba in zip(seeds, round_perm_probas, copy_perm_probas, masks_probas)))
    else:
        LIMIT = 1000
        per_trace = np.array([process_trace(seed, round_perm_proba, copy_perm_proba, masks_proba) for seed, round_perm_proba, copy_perm_proba, masks_proba in zip(seeds[:LIMIT], round_perm_probas[:LIMIT], copy_perm_probas[:LIMIT], masks_probas[:LIMIT])])

    with open(per_trace_filepath, "wb") as w:
        pic.dump(per_trace, w)

    return per_trace

def rws_process_trace(seed, rws_perm_proba: np.ndarray, rws_masks_proba: np.ndarray):
    assert rws_perm_proba.shape == (KEYROUND_WIDTH_B4,) and rws_masks_proba.shape == (KEYROUND_WIDTH_B4, NR_SHARES, len(KEY_ALPHABET))
    predicted_key = np.zeros((KEY_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.float64)

    rws_masks_proba = np.concatenate((rws_masks_proba, np.full((KEYROUND_WIDTH_B4 - rws_masks_proba.shape[0], NR_SHARES, len(KEY_ALPHABET)), np.log(1/len(KEY_ALPHABET)))), axis=0)
    
    predicted_values = np.stack([np.logaddexp.reduce([rws_masks_proba[:, 0, m] + rws_masks_proba[:, 1, (k - m) % 16] for m in KEY_ALPHABET], axis=0) for k in KEY_ALPHABET], axis=-1)

    predicted_keyround = np.zeros((KEYROUND_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.float64)
    for keyround_idx in range(KEYROUND_WIDTH_B4):
        for k in KEY_ALPHABET:
            predicted_keyround[keyround_idx, k] = np.logaddexp.reduce([predicted_values[(keyround_idx - rws_perm) % KEYROUND_WIDTH_B4, k] + rws_perm_proba[rws_perm] for rws_perm in range(KEYROUND_WIDTH_B4)], axis=0)

    indices, whitening = chacha_random_b4(seed)
    predicted_key[indices[:KEYROUND_WIDTH_B4]] = np.array([np.roll(pred, -whi) for pred, whi in zip(predicted_keyround, whitening)])

    return predicted_key

def rws_classifications_per_trace(per_trace_filepath: str, seeds: np.ndarray, rws_perm_probas: np.ndarray, rws_masks_probas: np.ndarray, parallel: bool = True):
    if parallel:
        per_trace = np.array(Parallel(n_jobs=-1)(delayed(rws_process_trace)(seed, rws_perm_proba, rws_masks_proba) for seed, rws_perm_proba, rws_masks_proba in zip(seeds, rws_perm_probas, rws_masks_probas)))
    else:
        LIMIT = 1000
        per_trace = np.array([rws_process_trace(seed, rws_perm_proba, rws_masks_proba) for seed, rws_perm_proba, rws_masks_proba in zip(seeds[:LIMIT], rws_perm_probas[:LIMIT], rws_masks_probas[:LIMIT])])

    with open(per_trace_filepath, "wb") as w:
        pic.dump(per_trace, w)

    return per_trace

def reconstruct_key(per_trace: np.ndarray):
    classifications_per_key_nibble = np.sum(per_trace, axis=0)
    recovered_key = np.argmax(classifications_per_key_nibble, axis=1)

    return recovered_key

def guessing_entropy(classifications_per_key_nibble: np.ndarray, key: np.ndarray):
    return np.argmax(np.argsort(classifications_per_key_nibble, axis=-1)[..., ::-1] == key[..., np.newaxis], axis=-1)
