import pickle as pic
from typing import List, Self, Tuple

from joblib import Parallel, delayed

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split, ShuffleSplit

from signal_strength import SignalStrength

from utils import *

class B4GaussianEstimator:
    """
    Class representing a template.
    fit() is used to estimate the distribution and predict() is used to apply its pdf.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Estimate the Gaussian distributions observed in X for each label in y.

        Args:
            X (np.ndarray): Traces dataset
            y (np.ndarray): Labels dataset

        Returns:
            Self: self
        """
        self.unique_y = np.unique(y)
        self.templates = np.zeros(len(self.unique_y), dtype=object)

        for l in range(len(self.unique_y)):
            data_lab = X[y == self.unique_y[l]]
            
            mean = np.mean(data_lab, axis=0)                               # Mean estimation
            cov  = np.nan_to_num(np.cov(data_lab, rowvar=False), nan=0.0)  # Covariance estimation

            self.templates[l] = multivariate_normal(mean, cov, allow_singular=True)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the Maximum-Likelihood estimator of y for each trace observed in X.

        Args:
            X (np.ndarray): Observed traces

        Returns:
            np.ndarray: Maximum-Likelihood estimator for each entry in X
        """
        return np.argmax(self.predict_proba(X), axis=1)            # Maximum Likelihood
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return log-probabilities of observing each trace in X for each label in y

        Args:
            X (np.ndarray): Observed traces

        Returns:
            np.ndarray: Log-probabilities of observing each trace for each label
        """
        return np.array([t.logpdf(X) for t in self.templates]).T   # Likelihoods

def b4_gridsearch_cv(sig_strength: SignalStrength, n_folds: int, train_size: int, num_features: List[int], traces_trainval: np.ndarray, labels_trainval: np.ndarray) -> Tuple[B4GaussianEstimator, int, float]:
    """Cross validated grid search on the number of features to choose for a given signal strength method. 

    Args:
        sig_strength (SignalStrength): Signal strength method
        n_folds (int): Number of folds in the cross-validation
        train_size (int): Train size
        num_features (List[int]): List of different values for the number of features
        traces_trainval (np.ndarray): Traces for training & validation
        labels_trainval (np.ndarray): Labels for training & validation

    Returns:
        Tuple[B4GaussianEstimator, int, float]: Best estimator, best number of features, best accuracy
    """
    rs = ShuffleSplit(n_splits=n_folds, train_size=train_size, random_state=0)
    results = np.zeros((len(num_features), n_folds), dtype=np.float64)
    models  = np.zeros((len(num_features), n_folds), dtype=object)
    # Cross-validation
    for cv, (train_index, val_index) in enumerate(rs.split(traces_trainval)):
        print(f"CV {cv}", end="\r")
        traces_train, traces_val = traces_trainval[train_index], traces_trainval[val_index]
        labels_train, labels_val = labels_trainval[train_index], labels_trainval[val_index]
        # Try different number of features
        for n, n_feat in enumerate(num_features):
            sig_strength.fit(traces_train, labels_train, n_feat)                               # Compute the signal strength on the given split
            features_train = sig_strength.transform(traces_train)                              # Feature selection
            gaussian_est = B4GaussianEstimator().fit(features_train, labels_train)             # Build templates

            features_val = sig_strength.transform(traces_val)                                  # Feature reduction on validation
            predicted = gaussian_est.predict(features_val)                                     # Prediction on validation
            results[n, cv] = np.count_nonzero(predicted == labels_val) / labels_val.shape[0]   # Accuracy
            models[n, cv] = gaussian_est

        del traces_train, traces_val, features_train, features_val
    for n, n_feat in enumerate(num_features):
        print(f"{sig_strength} [num features: {n_feat}]: {np.mean(results[n]):#.4g} ± {np.std(results[n]):#.4g} ({results[n]}).")
    print(f"-- BEST NUM_FEATURES: {num_features[np.argmax(np.mean(results, axis=1))]}--")

    return models[np.unravel_index(np.argmax(results), models.shape)], np.argmax(np.mean(results, axis=1)), np.max(np.mean(results, axis=1))

def get_all_labels(seeds: np.ndarray, key: np.ndarray, log_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get all masking shares in order where they appear in the trace and RSIs from the log output

    Args:
        seeds (np.ndarray): Seeds dataset
        key (np.ndarray): Key
        log_output (np.ndarray): Log output, containing the shares of the key and the RSIs 

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: RWS perm, RWS masks, round perm, block perm, round masks
    """
    keyshares, perms = log_output
    # RSIs are outputted in order in the log output
    rws_perms_labels = perms[:, 0]
    round_perms_labels = perms[:, 1]
    copy_perms_labels = perms[:, 2::8]

    # For masking shares, we need to compute where each key share appears in the trace, taking into account the shuffling
    rws_masks_labels = np.zeros((KEYROUND_WIDTH_B4, NR_SHARES, seeds.shape[0]), dtype=int)
    round_masks_labels = np.zeros((LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4, NR_SHARES, seeds.shape[0]), dtype=int)
    for i, (seed, key_share, rws_perm, round_perm, copy_perm) in enumerate(zip(seeds, keyshares, rws_perms_labels, round_perms_labels, copy_perms_labels)):
        indices, whitening = chacha_random_b4(seed)
        for observed_keyround_index in range(KEYROUND_WIDTH_B4):
            # RWS
            # Consider the shuffling
            manipulated_keyround_index = (rws_perm + observed_keyround_index) % KEYROUND_WIDTH_B4
            manipulated_key_index = indices[manipulated_keyround_index]

            assert np.sum(key_share[manipulated_key_index]) % 16 == key[manipulated_key_index]                                                     # Sanity check
            rws_masks_labels[observed_keyround_index, :-1, i] = key_share[manipulated_key_index, :-1]
            rws_masks_labels[observed_keyround_index, -1, i] = (key_share[manipulated_key_index, -1] + whitening[manipulated_keyround_index]) % 16 # Whitening is added in the last share

            # ROUNDS
            observed_round_idx = observed_keyround_index // BLOCK_WIDTH_B4
            observed_block_idx = observed_keyround_index % BLOCK_WIDTH_B4
            if not (EARLIEST_ROUND <= observed_round_idx < LATEST_ROUND):
                continue
            # Consider the shuffling
            copy_p = copy_perm[observed_round_idx]
            manipulated_keyround_index = ((round_perm + observed_round_idx) % (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4)) * BLOCK_WIDTH_B4 + ((copy_p + observed_block_idx) % BLOCK_WIDTH_B4)
            manipulated_key_index = indices[manipulated_keyround_index]

            assert np.sum(key_share[manipulated_key_index]) % 16 == key[manipulated_key_index]                                                     # Sanity check
            round_masks_labels[observed_round_idx, observed_block_idx, :-1, i] = key_share[manipulated_key_index, :-1]    
            round_masks_labels[observed_round_idx, observed_block_idx, -1, i] = (key_share[manipulated_key_index, -1] + whitening[manipulated_keyround_index]) % 16  # Whitening is added in the last share
    
    return rws_perms_labels, rws_masks_labels, round_perms_labels, copy_perms_labels.T, round_masks_labels

def classifications_per_trace(per_trace_filepath: str, seeds: np.ndarray, round_perm_probas: np.ndarray, copy_perm_probas: np.ndarray, round_masks_probas: np.ndarray, parallel: bool = True) -> np.ndarray:
    """Compute log-probabilities of the key for each observation and predictions on masking shares and RSIs.
    This function only focuses on the rounds.

    Args:
        per_trace_filepath (str): File path where to save the predictions of the key
        seeds (np.ndarray): Seeds dataset
        round_perm_probas (np.ndarray): Prediction's log-probabilities for the round perm
        copy_perm_probas (np.ndarray): Prediction's log-probabilities for the block perm
        round_masks_probas (np.ndarray): Prediction's log-probabilities for the round masks
        parallel (bool, optional): Whether or not to execute the function in parallel. Defaults to True.

    Returns:
        np.ndarray: For every observation, the log-probability of each key element
    """

    def process_trace(seed: str, round_perm_proba: np.ndarray, copy_perm_proba: np.ndarray, round_masks_proba: np.ndarray) -> np.ndarray:
        """Compute the log-probabilities of the key for the given observation and predictions

        Args:
            seed (str): The seed of the observation
            round_perm_proba (np.ndarray): The log-probabilities of the round perm of the observation
            copy_perm_proba (np.ndarray): The log-probabilities of the block perm of the observation
            round_masks_proba (np.ndarray): The log-probabilities of the round masks of the observation, either predicted together or independently

        Raises:
            ValueError: If the input is malformed

        Returns:
            np.ndarray: The log-probability of each key element for the given observation
        """
        # Sanity checks
        assert round_perm_proba.shape == (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4,) and copy_perm_proba.shape == (LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4) and (round_masks_proba.shape == (LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4, len(KEY_ALPHABET) ** NR_SHARES) or round_masks_proba.shape == (LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4, NR_SHARES, len(KEY_ALPHABET)))

        predicted_key = np.zeros((KEY_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.float64)

        if round_masks_proba.ndim == 3:
            # Masking shares were predicted together

            # Fill the array with uniform probability for unobserved rounds (i.e., those not between `EARLIEST_ROUND` and `LATEST_ROUND`)
            round_masks_proba = np.concatenate((np.full((EARLIEST_ROUND, BLOCK_WIDTH_B4, len(KEY_ALPHABET) ** NR_SHARES), np.log(1/(len(KEY_ALPHABET) ** NR_SHARES))), round_masks_proba, np.full((KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 - LATEST_ROUND, BLOCK_WIDTH_B4, len(KEY_ALPHABET) ** NR_SHARES), np.log(1/(len(KEY_ALPHABET) ** NR_SHARES)))), axis=0)

            # Probabilities for the keyround element, summing over all possibilities of the mask
            predicted_values = np.stack([np.logaddexp.reduce([round_masks_proba[:, :, len(KEY_ALPHABET) * m + ((k - m) % 16)] for m in KEY_ALPHABET], axis=0) for k in KEY_ALPHABET], axis=-1)
        elif round_masks_proba.ndim == 4:
            # Masking shares were predicted independently

            # Fill the array with uniform probability for unobserved rounds (i.e., those not between `EARLIEST_ROUND` and `LATEST_ROUND`)
            round_masks_proba = np.concatenate((np.full((EARLIEST_ROUND, BLOCK_WIDTH_B4, NR_SHARES, len(KEY_ALPHABET)), np.log(1/len(KEY_ALPHABET))), round_masks_proba, np.full((KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 - LATEST_ROUND, BLOCK_WIDTH_B4, NR_SHARES, len(KEY_ALPHABET)), np.log(1/len(KEY_ALPHABET)))), axis=0)

            # Probabilities for the keyround element, summing over all possibilities of the mask
            predicted_values = np.stack([np.logaddexp.reduce([round_masks_proba[:, :, 0, m] + round_masks_proba[:, :, 1, ((k - m) % 16)] for m in KEY_ALPHABET], axis=0) for k in KEY_ALPHABET], axis=-1)
        else:
            raise ValueError

        # Fill the array with uniform probability for unobserved rounds (i.e., those not between `EARLIEST_ROUND` and `LATEST_ROUND`)
        copy_perm_proba = np.concatenate((np.full((EARLIEST_ROUND, BLOCK_WIDTH_B4), np.log(1/BLOCK_WIDTH_B4)), copy_perm_proba, np.full((KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 - LATEST_ROUND, BLOCK_WIDTH_B4), np.log(1/BLOCK_WIDTH_B4))), axis=0)

        # Revert the block shuffling, summing over all its possibilities
        predicted_rounds = np.zeros((KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4, BLOCK_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.float64)
        for round_idx in range(KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4):
            for block_idx in range(BLOCK_WIDTH_B4):
                for k in KEY_ALPHABET:
                    predicted_rounds[round_idx, block_idx, k] = np.logaddexp.reduce([predicted_values[round_idx, (block_idx - copy_perm) % BLOCK_WIDTH_B4, k] + copy_perm_proba[round_idx, copy_perm] for copy_perm in range(BLOCK_WIDTH_B4)], axis=0)

        # Revert the round shuffling, summing over all its possibilities
        predicted_total = np.zeros((KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4, BLOCK_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.float64)
        for round_idx in range(KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4):
            for block_idx in range(BLOCK_WIDTH_B4):
                for k in KEY_ALPHABET:
                    predicted_total[round_idx, block_idx, k] = np.logaddexp.reduce([predicted_rounds[(round_idx - round_perm) % (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4), block_idx, k] + round_perm_proba[round_perm] for round_perm in range(KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4)], axis=0)

        predicted_keyround = np.concatenate(predicted_total, axis=0)                                                                       # From (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4, BLOCK_WIDTH_B4, ...) to (KEYROUND_WIDTH_B4, ...)
        indices, whitening = chacha_random_b4(seed)
        predicted_key[indices[:KEYROUND_WIDTH_B4]] = np.array([np.roll(pred, -whi) for pred, whi in zip(predicted_keyround, whitening)])   # Revert PRNG

        return predicted_key

    if parallel:
        per_trace = np.array(Parallel(n_jobs=-1)(delayed(process_trace)(seed, round_perm_proba, copy_perm_proba, round_masks_proba) for seed, round_perm_proba, copy_perm_proba, round_masks_proba in zip(seeds, round_perm_probas, copy_perm_probas, round_masks_probas)))
    else:
        LIMIT = 1000
        per_trace = np.array([process_trace(seed, round_perm_proba, copy_perm_proba, round_masks_proba) for seed, round_perm_proba, copy_perm_proba, round_masks_proba in zip(seeds[:LIMIT], round_perm_probas[:LIMIT], copy_perm_probas[:LIMIT], round_masks_probas[:LIMIT])])

    with open(per_trace_filepath, "wb") as w:
        pic.dump(per_trace, w)

    return per_trace

def rws_classifications_per_trace(per_trace_filepath: str, seeds: np.ndarray, rws_perm_probas: np.ndarray, rws_masks_probas: np.ndarray, parallel: bool = True) -> np.ndarray:
    """Compute log-probabilities of the key for each observation and predictions on masking shares and RSIs.
    This function only focuses on RWS.

    Args:
        per_trace_filepath (str): File path where to save the predictions of the key
        seeds (np.ndarray): Seeds dataset
        rws_perm_probas (np.ndarray): Prediction's log-probabilities for the RWS perm
        rws_masks_probas (np.ndarray): Prediction's log-probabilities for the RWS masks
        parallel (bool, optional): Whether or not to execute the function in parallel. Defaults to True.

    Returns:
        np.ndarray: For every observation, the log-probability of each key element
    """

    def rws_process_trace(seed: str, rws_perm_proba: np.ndarray, rws_masks_proba: np.ndarray) -> np.ndarray:
        """Compute the log-probabilities of the key for the given observation and predictions

        Args:
            seed (str): The seed of the observation
            round_perm_proba (np.ndarray): The log-probabilities of the round perm of the observation
            copy_perm_proba (np.ndarray): The log-probabilities of the block perm of the observation
            rws_masks_proba (np.ndarray): The log-probabilities of the round masks of the observation, either predicted together or independently

        Returns:
            np.ndarray: The log-probability of each key element for the given observation
        """
        # Sanity checks
        assert rws_perm_proba.shape == (KEYROUND_WIDTH_B4,) and rws_masks_proba.shape == (KEYROUND_WIDTH_B4, NR_SHARES, len(KEY_ALPHABET))

        predicted_key = np.zeros((KEY_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.float64)

        # Fill the array with uniform probability for unobserved RWS parts (typically, we focus on a single quarter of it)
        rws_masks_proba = np.concatenate((rws_masks_proba, np.full((KEYROUND_WIDTH_B4 - rws_masks_proba.shape[0], NR_SHARES, len(KEY_ALPHABET)), np.log(1/len(KEY_ALPHABET)))), axis=0)
        
        # Probabilities for the keyround element, summing over all possibilities of the mask
        predicted_values = np.stack([np.logaddexp.reduce([rws_masks_proba[:, 0, m] + rws_masks_proba[:, 1, (k - m) % 16] for m in KEY_ALPHABET], axis=0) for k in KEY_ALPHABET], axis=-1)

        # Revert the RWS shuffling, summing over all its possibilities
        predicted_keyround = np.zeros((KEYROUND_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.float64)
        for keyround_idx in range(KEYROUND_WIDTH_B4):
            for k in KEY_ALPHABET:
                predicted_keyround[keyround_idx, k] = np.logaddexp.reduce([predicted_values[(keyround_idx - rws_perm) % KEYROUND_WIDTH_B4, k] + rws_perm_proba[rws_perm] for rws_perm in range(KEYROUND_WIDTH_B4)], axis=0)

        indices, whitening = chacha_random_b4(seed)
        predicted_key[indices[:KEYROUND_WIDTH_B4]] = np.array([np.roll(pred, -whi) for pred, whi in zip(predicted_keyround, whitening)])   # Revert PRNG

        return predicted_key

    if parallel:
        per_trace = np.array(Parallel(n_jobs=-1)(delayed(rws_process_trace)(seed, rws_perm_proba, rws_masks_proba) for seed, rws_perm_proba, rws_masks_proba in zip(seeds, rws_perm_probas, rws_masks_probas)))
    else:
        LIMIT = 1000
        per_trace = np.array([rws_process_trace(seed, rws_perm_proba, rws_masks_proba) for seed, rws_perm_proba, rws_masks_proba in zip(seeds[:LIMIT], rws_perm_probas[:LIMIT], rws_masks_probas[:LIMIT])])

    with open(per_trace_filepath, "wb") as w:
        pic.dump(per_trace, w)

    return per_trace

def reconstruct_key(per_trace: np.ndarray) -> np.ndarray:
    """From all the log-probabilities of each observation, we select the Maximum-Likelihood one (sum them together and select the highest)

    Args:
        per_trace (np.ndarray): For every observation, the log-probability of each key element

    Returns:
        np.ndarray: Maximum-Likelihood estimator of each key element
    """
    classifications_per_key_nibble = np.sum(per_trace, axis=0)
    recovered_key = np.argmax(classifications_per_key_nibble, axis=1)

    return recovered_key

def guessing_entropy(classifications_per_key_nibble: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Empirically computes the guessing entropy of the predictions

    Args:
        classifications_per_key_nibble (np.ndarray): The sum of log-probabilities of each observation, for each key element
        key (np.ndarray): The real key

    Returns:
        np.ndarray: The guessing entropy, i.e., the rank of every key element
    """
    return np.argmax(np.argsort(classifications_per_key_nibble, axis=-1)[..., ::-1] == key[..., np.newaxis], axis=-1)
