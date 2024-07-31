import sys

import numpy as np

from utils import *
from signal_strength import SIGNAL_STRENGTHS_METHODS
from data_loader import EntireTraceIterator

"""
Script to be run multiple times in parallel to go faster (instead of relying on a single notebook).
It computes the SOST function for a certain device/key pair given through the command line.
"""

def sost(filename: str, traces: np.ndarray, labels: np.ndarray):
    """Computes (and stores) the SOST from the traces and labels

    Args:
        filename (str): File path to store the SOST
        traces (np.ndarray): Traces dataset
        labels (np.ndarray): Labels dataset
    """
    signal_strength = SIGNAL_STRENGTHS_METHODS["SOST"](filename)
    signal_strength.fit(traces, labels, 0)

if __name__ == '__main__':
    traces_path = "..\\acquisition\\patent_countermeasure_v2_25000\\carto_eB4-Rnd-3-WhiteningAndFullFilter-Patented-4-Devices-3-Keys.mat"
    key_path = "..\\acquisition\\patent_countermeasure_v2_25000\\carto_eB4-Rnd-3-WhiteningAndFullFilter-Patented-4-Devices-3-Keys.log"
    TRACE_SIZE = 250_000
    data_loader = EntireTraceIterator(traces_path, key_path, trace_size=TRACE_SIZE, traces_per_division=150_000, parse_output="delays")
    seeds_11, seeds_12, seeds_13, seeds_21, seeds_31, seeds_41, labels_11, labels_12, labels_13, labels_21, labels_31, labels_41, traces_11, traces_12, traces_13, traces_21, traces_31, traces_41, keys_11, keys_12, keys_13, keys_21, keys_31, keys_41, delays_11, delays_12, delays_13, delays_21, delays_31, delays_41 = data_loader.full(150_000)

    device = int(sys.argv[1])
    key = int(sys.argv[2])

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
            raise ValueError
    elif device == 2:
        if key == 1:
            traces = traces_21
            labels = labels_21
        else:
            raise ValueError
    elif device == 3:
        if key == 1:
            traces = traces_31
            labels = labels_31
        else:
            raise ValueError
    elif device == 4:
        if key == 1:
            traces = traces_41
            labels = labels_41
        else:
            raise ValueError
    else:
        raise ValueError

    del seeds_11, seeds_12, seeds_13, seeds_21, seeds_31, seeds_41, labels_11, labels_12, labels_13, labels_21, labels_31, labels_41, traces_11, traces_12, traces_13, traces_21, traces_31, traces_41, keys_11, keys_12, keys_13, keys_21, keys_31, keys_41, delays_11, delays_12, delays_13, delays_21, delays_31, delays_41

    # Compute the SOST for every keyround element
    for keyround_idx in range(KEYROUND_WIDTH_B4):
        print(keyround_idx, end="\r")
        for share_idx in range(NR_SHARES):
            sost(f"leakage_points_v2_rws_1st_round/f_rws_{device}_{key}_{keyround_idx}_sost.pic", traces, labels[:, keyround_idx])
