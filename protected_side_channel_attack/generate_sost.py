import sys

from utils import *
from templates import *
from signal_strength import SIGNAL_STRENGTHS_METHODS

from itertools import product

from joblib import Parallel, delayed

def rws_perm():
    signal_strength = SIGNAL_STRENGTHS_METHODS["SOST"]("./leakage_points_haar_2/f_rws_perms_sost.pic")
    signal_strength.fit(wavelets, rws_perms_labels, 0)

def rws_masks(keyround_idx, share_idx):
    signal_strength = SIGNAL_STRENGTHS_METHODS["SOST"](f"./leakage_points_haar_2/f_rws_masks_{keyround_idx}_{share_idx}_sost.pic")
    signal_strength.fit(wavelets, rws_masks_labels[keyround_idx, share_idx], 0)

def round_perm():
    signal_strength = SIGNAL_STRENGTHS_METHODS["SOST"]("./leakage_points_haar_2/f_round_perms_sost.pic")
    signal_strength.fit(wavelets, round_perms_labels, 0)

def copy_perms(round_idx):
    signal_strength = SIGNAL_STRENGTHS_METHODS["SOST"](f"./leakage_points_haar_2/f_copy_perms_{round_idx}_sost.pic")
    signal_strength.fit(wavelets, copy_perms_labels[round_idx], 0)

def round_masks(round_idx, block_idx, share_idx):
    signal_strength = SIGNAL_STRENGTHS_METHODS["SOST"](f"./leakage_points_haar_2/f_round_masks_{round_idx}_{block_idx}_{share_idx}_sost.pic")
    signal_strength.fit(wavelets, round_masks_labels[round_idx, block_idx, share_idx], 0)

if __name__ == '__main__':
    with open("wavelets_haar_2.pic", "rb") as r:
        wavelets = pic.load(r)
    with open("labels.pic", "rb") as r:
        rws_perms_labels, round_perms_labels, copy_perms_labels, rws_masks_labels, round_masks_labels = pic.load(r)

    #n_jobs = 4
    #_ = Parallel(n_jobs=n_jobs)(delayed(copy_perms)(round_idx) for round_idx in range(EARLIEST_ROUND, LATEST_ROUND))
    #_ = Parallel(n_jobs=n_jobs)(delayed(rws_masks)(round_idx, share_idx) for round_idx, share_idx in product(range(KEYROUND_WIDTH_B4), range(NR_SHARES)))
    #_ = Parallel(n_jobs=n_jobs)(delayed(round_masks)(round_idx, block_idx, share_idx) for round_idx, block_idx, share_idx in product(range(EARLIEST_ROUND, LATEST_ROUND), range(BLOCK_WIDTH_B4), range(NR_SHARES)))

    arg = sys.argv[1]

    if arg == "rws_perm":
        rws_perm()
    elif arg == "rws_masks":
        for keyround_idx in range(KEYROUND_WIDTH_B4):
            print(keyround_idx, end="\r")
            for share_idx in range(NR_SHARES):
                rws_masks(keyround_idx, share_idx)
    elif arg == "round_perm":
        round_perm()
    elif arg == "copy_perm":
        for round_idx in range(EARLIEST_ROUND, LATEST_ROUND):
            print(round_idx, end="\r")
            copy_perms(round_idx)
    elif arg == "round_masks":
        for round_idx in range(EARLIEST_ROUND, LATEST_ROUND):
            for block_idx in range(BLOCK_WIDTH_B4):
                print(round_idx * BLOCK_WIDTH_B4 + block_idx, end="\r")
                for share_idx in range(NR_SHARES):
                    round_masks(round_idx, block_idx, share_idx)