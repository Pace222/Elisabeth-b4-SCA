import numpy as np
import pickle as pic
from itertools import product

from joblib import Parallel, delayed

from utils import *
from templates import *
from signal_strength import SIGNAL_STRENGTHS_METHODS

def process_trace(seed, round_perm_proba, copy_perm_proba, masks_proba, round_keep_only: np.ndarray, copy_keep_only: np.ndarray, masks_keep_only: np.ndarray):
    assert round_perm_proba.shape[0] == KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 and copy_perm_proba.shape[1] == BLOCK_WIDTH_B4

    classifications_per_key_nibble = np.zeros((KEY_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.longfloat)
    if np.argmax(round_perm_proba) not in round_keep_only:
        return classifications_per_key_nibble

    indices, whitening = chacha_random_b4(seed)

    classifications_per_keyround_nibble = np.zeros((KEYROUND_WIDTH_B4, len(KEY_ALPHABET)), dtype=np.longfloat)
    for target_keyround_index in range(KEYROUND_WIDTH_B4):
        target_round_idx = target_keyround_index // BLOCK_WIDTH_B4
        target_block_idx = target_keyround_index % BLOCK_WIDTH_B4

        for target_witness in KEY_ALPHABET:
            classifications_hypotheses = np.full((LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4, len(KEY_ALPHABET) ** (NR_SHARES - 1)), -np.inf, dtype=np.longfloat)

            for round_idx in range(EARLIEST_ROUND, LATEST_ROUND):
                round_proba = round_perm_proba[(target_round_idx - round_idx) % (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4)]
                
                if np.argmax(copy_perm_proba[round_idx]) not in copy_keep_only[round_idx]:
                    continue

                for block_idx in range(BLOCK_WIDTH_B4):
                    copy_proba = copy_perm_proba[round_idx, (target_block_idx - block_idx) % BLOCK_WIDTH_B4]

                    if np.argmax(masks_proba[round_idx, block_idx]) not in masks_keep_only[round_idx, block_idx]:
                        continue

                    for m, mask in enumerate(product(KEY_ALPHABET, repeat=NR_SHARES-1)):
                        masked_value = (target_witness + whitening[target_keyround_index] - np.sum(mask)) % 16
                        probas = masks_proba[round_idx, block_idx, len(KEY_ALPHABET) * m + masked_value]

                        classifications_hypotheses[round_idx, block_idx, m] = round_proba + copy_proba + probas
            classifications_per_keyround_nibble[target_keyround_index, target_witness] = np.nan_to_num(np.logaddexp.reduce(classifications_hypotheses, axis=None), neginf=0)

    classifications_per_key_nibble[indices[:KEYROUND_WIDTH_B4]] = classifications_per_keyround_nibble
    return classifications_per_key_nibble

def reconstruct_key(seeds: np.ndarray, round_perm_probas: np.ndarray, copy_perm_probas: np.ndarray, masks_probas: np.ndarray, round_keep_only: np.ndarray = None, copy_keep_only: np.ndarray = None, masks_keep_only: np.ndarray = None):
    if round_keep_only is None:
        round_keep_only = np.arange(round_perm_probas.shape[1])
    if copy_keep_only is None:
        copy_keep_only = np.tile(np.arange(copy_perm_probas.shape[2]), copy_perm_probas.shape[1:-1] + (1,))
    if masks_keep_only is None:
        masks_keep_only = np.tile(np.arange(masks_probas.shape[3]), masks_probas.shape[1:-1] + (1,))

    #LIMIT = 5
    #classifications_per_trace = [process_trace(seed, round_perm_proba, copy_perm_proba, masks_proba, round_keep_only, copy_keep_only, masks_keep_only) for seed, round_perm_proba, copy_perm_proba, masks_proba in zip(seeds[:LIMIT], round_perm_probas[:LIMIT], copy_perm_probas[:LIMIT], masks_probas[:LIMIT])]
    classifications_per_trace = Parallel(n_jobs=-1)(delayed(process_trace)(seed, round_perm_proba, copy_perm_proba, masks_proba, round_keep_only, copy_keep_only, masks_keep_only) for seed, round_perm_proba, copy_perm_proba, masks_proba in zip(seeds, round_perm_probas, copy_perm_probas, masks_probas))
    classifications_per_key_nibble = np.sum(classifications_per_trace, axis=0)

    recovered_key = np.argmax(classifications_per_key_nibble, axis=1)
    with open("classifications_per_key_nibble.pic", "wb") as w:
        pic.dump(classifications_per_key_nibble, w)
    with open("recovered_key.pic", "wb") as w:
        pic.dump(recovered_key, w)
    return recovered_key

try:
    with open(f"predicted_probas_sost_plus_pca_5_features.pic", "rb") as r:
        key, seeds_test, predicted_round_perm_probas, predicted_copy_perm_probas, predicted_masks_probas = pic.load(r)
    recovered_key = reconstruct_key(seeds_test, predicted_round_perm_probas, predicted_copy_perm_probas, predicted_masks_probas)

    print(f"Recovered {np.count_nonzero(recovered_key == key) / KEY_WIDTH_B4:.2%} of the key.")
    exit(0)
except FileNotFoundError:
    pass

with open("full_data.pic", "rb") as r:
    seeds_total, traces_total, key, output_total, keyshares_total, perms_total = pic.load(r)

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
train_sizes = [250_000]
val_sizes = [5_134]
num_features = [200]

seeds_trainval, seeds_test, traces_trainval, traces_test, round_perms_trainval,round_perms_test, copy_perms_trainval, copy_perms_test, masks_trainval, masks_test = b4_train_test_split(seeds_total, traces_total, round_perms_labels, copy_perms_labels.T, np.transpose(masks_labels, (3, 0, 1, 2)), train_size=train_sizes[0], val_sizes=val_sizes)
copy_perms_trainval, copy_perms_test = copy_perms_trainval.T, copy_perms_test.T
masks_trainval, masks_test = np.transpose(masks_trainval, (1, 2, 3, 0)), np.transpose(masks_test, (1, 2, 3, 0))

sig_strength_round_perms = SIGNAL_STRENGTHS_METHODS["SOST"](f"f_{train_sizes[0] + val_sizes[0]}_round_perms_best.pic")
sig_strength_round_perms_pca = SIGNAL_STRENGTHS_METHODS["PCA"](f"f_{train_sizes[0] + val_sizes[0]}_round_perms_best_after_pca_5.pic")
best_round_perms_model = b4_train(sig_strength_round_perms_pca, 5, sig_strength_round_perms.fit(traces_trainval, round_perms_trainval, 200).transform(traces_trainval), round_perms_trainval)

sig_strength_copy_perms = np.zeros(copy_perms_labels.shape[0], dtype=SignalStrength)
sig_strength_copy_perms_pca = np.zeros(copy_perms_labels.shape[0], dtype=SignalStrength)
best_copy_perms_model = np.zeros(copy_perms_labels.shape[0], dtype=B4GaussianEstimator)
for copy_index in range(copy_perms_labels.shape[0]):
    sig_strength_copy_perms[copy_index] = SIGNAL_STRENGTHS_METHODS["SOST"](f"f_{train_sizes[0] + val_sizes[0]}_copy_perms_{copy_index}_best.pic")
    sig_strength_copy_perms_pca[copy_index] = SIGNAL_STRENGTHS_METHODS["PCA"](f"f_{train_sizes[0] + val_sizes[0]}_copy_perms_{copy_index}_best_after_pca_5.pic")
    best_copy_perms_model[copy_index] = b4_train(sig_strength_copy_perms_pca[copy_index], 5, sig_strength_copy_perms[copy_index].fit(traces_trainval, copy_perms_trainval[copy_index], 200).transform(traces_trainval), copy_perms_trainval[copy_index])

sig_strength_masks = np.zeros((masks_labels.shape[0], masks_labels.shape[1]), dtype=SignalStrength)
sig_strength_masks_pca = np.zeros((masks_labels.shape[0], masks_labels.shape[1]), dtype=SignalStrength)
best_masks_models = np.zeros((masks_labels.shape[0], masks_labels.shape[1]), dtype=B4GaussianEstimator)
for round_idx in range(masks_labels.shape[0]):
    for block_idx in range(masks_labels.shape[1]):
        sig_strength_masks[round_idx, block_idx] = SIGNAL_STRENGTHS_METHODS["SOST"](f"f_{train_sizes[0] + val_sizes[0]}_masks_{round_idx}_{block_idx}_best.pic")
        sig_strength_masks_pca[round_idx, block_idx] = SIGNAL_STRENGTHS_METHODS["PCA"](f"f_{train_sizes[0] + val_sizes[0]}_masks_{round_idx}_{block_idx}_best_after_pca_5.pic")
        best_masks_models[round_idx, block_idx] = b4_train(sig_strength_masks_pca[round_idx, block_idx], 5, sig_strength_masks[round_idx, block_idx].fit(traces_trainval, masks_trainval[round_idx, block_idx, 1], 20).transform(traces_trainval), masks_trainval[round_idx, block_idx].T, masking=True)

predicted_round_perm_probas = best_round_perms_model.predict_proba(sig_strength_round_perms_pca.transform(sig_strength_round_perms.transform(traces_test)))

predicted_copy_perm_probas = np.zeros((seeds_test.shape[0], LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4))
for copy_index in range(copy_perms_labels.shape[0]):
    predicted_copy_perm_probas[:, copy_index, :] = best_copy_perms_model[copy_index].predict_proba(sig_strength_copy_perms_pca[copy_index].transform(sig_strength_copy_perms[copy_index].transform(traces_test)))

predicted_masks_probas = np.zeros((seeds_test.shape[0], LATEST_ROUND - EARLIEST_ROUND, BLOCK_WIDTH_B4, len(KEY_ALPHABET) ** NR_SHARES))
for round_idx in range(masks_labels.shape[0]):
    for block_idx in range(masks_labels.shape[1]):
        predicted_masks_probas[:, round_idx, block_idx, :] = best_masks_models[round_idx, block_idx].predict_proba(sig_strength_masks_pca[round_idx, block_idx].transform(sig_strength_masks[round_idx, block_idx].transform(traces_test)))
#        for share_idx in range(masks_labels.shape[2]):
#            predicted_masks_probas[:, round_idx, block_idx, share_idx, :] = best_masks_models[round_idx, block_idx, share_idx].predict_proba(sig_strength_masks[round_idx, block_idx, share_idx].transform(traces_test))

del seeds_trainval, traces_trainval, traces_test, round_perms_trainval,round_perms_test, copy_perms_trainval, copy_perms_test, masks_trainval, masks_test
del seeds_total, traces_total, output_total
with open(f"predicted_probas_sost_plus_pca_5_features_{train_sizes[0] + val_sizes[0]}_{predicted_round_perm_probas.shape[0]}.pic", "wb") as w:
    pic.dump((key, seeds_test, predicted_round_perm_probas, predicted_copy_perm_probas, predicted_masks_probas), w)
recovered_key = reconstruct_key(seeds_test, predicted_round_perm_probas, predicted_copy_perm_probas, predicted_masks_probas)

print(f"Recovered {np.count_nonzero(recovered_key == key) / KEY_WIDTH_B4:.2%} of the key.")
