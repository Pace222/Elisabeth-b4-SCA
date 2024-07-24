import sys

from utils import *
from templates import *
from signal_strength import SIGNAL_STRENGTHS_METHODS
from data_loader import EntireTraceIterator

from itertools import product

from joblib import Parallel, delayed


def rws(filename, traces, labels):
    signal_strength = SIGNAL_STRENGTHS_METHODS["SOST"](filename)
    signal_strength.fit(traces, labels, 0)

def rounds(filename, traces, labels):
    signal_strength = SIGNAL_STRENGTHS_METHODS["SOST"](filename)
    signal_strength.fit(traces, labels, 0)

if __name__ == '__main__':
    traces_path = "..\\acquisition\\patent_countermeasure_v2_25000\\carto_eB4-Rnd-3-WhiteningAndFullFilter-Patented-4-Devices-3-Keys.mat"
    key_path = "..\\acquisition\\patent_countermeasure_v2_25000\\carto_eB4-Rnd-3-WhiteningAndFullFilter-Patented-4-Devices-3-Keys.log"
    TRACE_SIZE = 250_000
    data_loader = EntireTraceIterator(traces_path, key_path, trace_size=TRACE_SIZE, traces_per_division=150_000, parse_output="delays")
    seeds_11, seeds_12, seeds_13, seeds_21, seeds_31, seeds_41, labels_11, labels_12, labels_13, labels_21, labels_31, labels_41, traces_11, traces_12, traces_13, traces_21, traces_31, traces_41, keys_11, keys_12, keys_13, keys_21, keys_31, keys_41, delays_11, delays_12, delays_13, delays_21, delays_31, delays_41 = data_loader.full(150_000)
    #n_jobs = 4
    #_ = Parallel(n_jobs=n_jobs)(delayed(copy_perms)(round_idx) for round_idx in range(EARLIEST_ROUND, LATEST_ROUND))
    #_ = Parallel(n_jobs=n_jobs)(delayed(rws_masks)(round_idx, share_idx) for round_idx, share_idx in product(range(KEYROUND_WIDTH_B4), range(NR_SHARES)))
    #_ = Parallel(n_jobs=n_jobs)(delayed(round_masks)(round_idx, block_idx, share_idx) for round_idx, block_idx, share_idx in product(range(EARLIEST_ROUND, LATEST_ROUND), range(BLOCK_WIDTH_B4), range(NR_SHARES)))

    mode = sys.argv[1]
    device = int(sys.argv[2])
    key = int(sys.argv[3])

    if device == 1:
        if key == 1:
            traces = traces_11
            labels = labels_11
        elif key == 2:
            traces = traces_12
            labels = labels_12
        elif key == 3:
            traces = traces_13
            labels = labels_13
        else:
            raise ValueError()
    elif device == 2:
        if key == 1:
            traces = traces_21
            labels = labels_21
        else:
            raise ValueError()
    elif device == 3:
        if key == 1:
            traces = traces_31
            labels = labels_31
        else:
            raise ValueError()
    elif device == 4:
        if key == 1:
            traces = traces_41
            labels = labels_41
        else:
            raise ValueError()
    else:
        raise ValueError()

    del seeds_11, seeds_12, seeds_13, seeds_21, seeds_31, seeds_41, labels_11, labels_12, labels_13, labels_21, labels_31, labels_41, traces_11, traces_12, traces_13, traces_21, traces_31, traces_41, keys_11, keys_12, keys_13, keys_21, keys_31, keys_41, delays_11, delays_12, delays_13, delays_21, delays_31, delays_41

    if mode == "rws":
        for keyround_idx in range(KEYROUND_WIDTH_B4):
            print(keyround_idx, end="\r")
            for share_idx in range(NR_SHARES):
                rws(f"./leakage_points_v2_rws_1st_round/f_rws_{device}_{key}_{keyround_idx}_sost.pic", traces, labels[:, keyround_idx])
    elif mode == "round":
        for round_idx in range(EARLIEST_ROUND, LATEST_ROUND):
            for block_idx in range(BLOCK_WIDTH_B4):
                keyround_idx = round_idx * BLOCK_WIDTH_B4 + block_idx
                print(keyround_idx, end="\r")

                rounds(f"./leakage_points_v2_rws_1st_round/f_round_{device}_{key}_{round_idx}_{block_idx}_sost.pic", traces, labels[:, keyround_idx])
    else:
        raise ValueError()