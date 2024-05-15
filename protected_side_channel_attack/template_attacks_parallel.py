import numpy as np
import pickle as pic
from itertools import product

from joblib import Parallel, delayed

from utils import *
from templates import *
from signal_strength import SIGNAL_STRENGTHS_METHODS

EARLIEST_ROUND = 0
LATEST_ROUND = 1
KEY_ALPHABET = list(range(16))

traces_path = "..\\acquisition\\395134_maskshuffle\\carto_eB4-Rnd-3-WhiteningAndFullFilter-Masking-Shuffling.mat"
key_path = "..\\acquisition\\395134_maskshuffle\\carto_eB4-Rnd-3-WhiteningAndFullFilter-Masking-Shuffling.log"

NUM_TRACES = 395_134
TRACE_SIZE = 80_000
data_loader = EntireTraceIterator(traces_path, key_path, nr_populations=1, nr_scenarios=1, trace_size=TRACE_SIZE, traces_per_division=NUM_TRACES, parse_output="keyshares+perms")

seeds_total, traces_total, key, output_total = data_loader((0,), (0,)).full(NUM_TRACES)
keyshares_total = output_total[0]
perms_total = output_total[1]

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

round_perms_labels = perms_total[:, 1]
copy_perms_labels = perms_total[:, 2:3:1].T
masks_labels = get_masks_labels(seeds_total, key, keyshares_total, round_perms_labels, copy_perms_labels.T)

n_folds = 2
train_sizes = [350_000]
val_sizes = [5_134]
num_features = [200]

seeds_trainval, seeds_test, traces_trainval, traces_test, round_perms_trainval,round_perms_test, copy_perms_trainval, copy_perms_test, masks_trainval, masks_test = b4_train_test_split(seeds_total, traces_total, round_perms_labels, copy_perms_labels.T, np.transpose(masks_labels, (3, 0, 1, 2)), train_size=train_sizes[0], val_sizes=val_sizes)
copy_perms_trainval, copy_perms_test = copy_perms_trainval.T, copy_perms_test.T
masks_trainval, masks_test = np.transpose(masks_trainval, (1, 2, 3, 0)), np.transpose(masks_test, (1, 2, 3, 0))

sig_strength_round_perms = SIGNAL_STRENGTHS_METHODS["SOST"]("f_355134_round_perms_best.pic")
best_round_perms_model = b4_train(sig_strength_round_perms, 200, traces_trainval, round_perms_trainval)

sig_strength_copy_perms = np.zeros(copy_perms_labels.shape[0], dtype=SignalStrength)
best_copy_perms_model = np.zeros(copy_perms_labels.shape[0], dtype=B4GaussianEstimator)
for copy_index in range(copy_perms_labels.shape[0]):
    sig_strength_copy_perms[copy_index] = SIGNAL_STRENGTHS_METHODS["SOST"](f"f_355134_copy_perms_{copy_index}_best.pic")
    best_copy_perms_model[copy_index] = b4_train(sig_strength_copy_perms[copy_index], 200, traces_trainval, copy_perms_trainval[copy_index])

sig_strength_masks = np.zeros((masks_labels.shape[0], masks_labels.shape[1]), dtype=SignalStrength)
best_masks_models = np.zeros((masks_labels.shape[0], masks_labels.shape[1]), dtype=B4GaussianEstimator)
for round_idx in range(masks_labels.shape[0]):
    for block_idx in range(masks_labels.shape[1]):
        sig_strength_masks[round_idx, block_idx] = SIGNAL_STRENGTHS_METHODS["SOST"](f"f_355134_masks_{round_idx}_{block_idx}_best.pic")
        best_masks_models[round_idx, block_idx] = b4_train(sig_strength_masks[round_idx, block_idx], 20, traces_trainval, masks_trainval[round_idx, block_idx].T, masking=True)

predicted_round_perm_probas = best_round_perms_model.predict_proba(sig_strength_round_perms.transform(traces_test))

predicted_copy_perm_probas = np.zeros((seeds_test.shape[0], LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4))
for copy_index in range(copy_perms_labels.shape[0]):
    predicted_copy_perm_probas[:, copy_index, :] = best_copy_perms_model[copy_index].predict_proba(sig_strength_copy_perms[copy_index].transform(traces_test))

predicted_masks_probas = np.zeros((seeds_test.shape[0], LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4, len(KEY_ALPHABET) ** NR_SHARES))
for round_idx in range(masks_labels.shape[0]):
    for block_idx in range(masks_labels.shape[1]):
        predicted_masks_probas[:, round_idx, block_idx, :] = best_masks_models[round_idx, block_idx].predict_proba(sig_strength_masks[round_idx, block_idx].transform(traces_test))
#        for share_idx in range(masks_labels.shape[2]):
#            predicted_masks_probas[:, round_idx, block_idx, share_idx, :] = best_masks_models[round_idx, block_idx, share_idx].predict_proba(sig_strength_masks[round_idx, block_idx, share_idx].transform(traces_test))

def process_trace(seed, round_perm_proba, copy_perm_proba, masks_proba):
    assert round_perm_proba.shape[0] == KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 and copy_perm_proba.shape[1] == BLOCK_WIDTH_B4
    classifications_per_key_nibble = np.zeros((KEY_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.longfloat)

    indices, whitening = chacha_random_b4(seed)
    round_perms = round_perm_proba[:]
    for round_idx in range(EARLIEST_ROUND, LATEST_ROUND):
        for block_idx in range(BLOCK_WIDTH_B4):
            classifications = np.full((KEY_WIDTH_B4, len(KEY_ALPHABET), len(KEY_ALPHABET) ** (NR_SHARES - 1)), -np.inf, dtype=np.longfloat)

            copy_perms = copy_perm_proba[round_idx]
            for round_perm, round_proba in enumerate(round_perms):
                for copy_perm, copy_proba in enumerate(copy_perms):
                    permuted_keyround_index = ((round_perm + round_idx) % (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4)) * BLOCK_WIDTH_B4 + ((copy_perm + block_idx) % BLOCK_WIDTH_B4)
                    key_index = indices[permuted_keyround_index]

                    for m, mask_shares in enumerate(product(KEY_ALPHABET, repeat=NR_SHARES)):
                        witness = (np.sum(mask_shares) - whitening[permuted_keyround_index]) % 16
                        probas = masks_proba[round_idx, block_idx, m]

                        classifications[key_index, witness, m // len(KEY_ALPHABET)] = round_proba + copy_proba + probas
            classifications_per_key_nibble += np.nan_to_num(np.logaddexp.reduce(classifications, axis=2), neginf=0)

    return classifications_per_key_nibble

def reconstruct_key(seeds: np.ndarray, round_perm_probas: np.ndarray, copy_perm_probas: np.ndarray, masks_probas: np.ndarray):
    classifications_per_trace = Parallel(n_jobs=-1)(delayed(process_trace)(seed, round_perm_proba, copy_perm_proba, masks_proba) for seed, round_perm_proba, copy_perm_proba, masks_proba in zip(seeds, round_perm_probas, copy_perm_probas, masks_probas))
    classifications_per_key_nibble = np.sum(classifications_per_trace, axis=0)

    recovered_key = np.argmax(classifications_per_key_nibble, axis=1)
    with open("classifications_per_key_nibble.pic", "wb") as w:
        pic.dump(classifications_per_key_nibble, w)
    with open("recovered_key.pic", "wb") as w:
        pic.dump(recovered_key, w)
    return recovered_key

del seeds_trainval, traces_trainval, traces_test, round_perms_trainval,round_perms_test, copy_perms_trainval, copy_perms_test, masks_trainval, masks_test
del seeds_total, traces_total, output_total
recovered_key = reconstruct_key(seeds_test, predicted_round_perm_probas, predicted_copy_perm_probas, predicted_masks_probas)

print(f"Recovered {np.count_nonzero(recovered_key == key) / KEY_WIDTH_B4:.2%} of the key.")
