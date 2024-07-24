from typing import List, Tuple
import pickle as pic
from collections import Counter

import numpy as np
#import cupy as np
import matplotlib.pyplot as plt
import scipy.io as sio


import log_parser

KEYROUND_WIDTH_4 = 60
KEYROUND_WIDTH_B4 = 98

KEY_WIDTH_4 = 256
KEY_WIDTH_B4 = 512

BLOCK_WIDTH_4 = 5
BLOCK_WIDTH_B4 = 7

HW = [bin(n).count("1") for n in range(0, 128)]
HD = [[HW[n1 ^ n2] for n2 in range(0, 128)] for n1 in range(0, 128)]

def corr_coef(hypotheses, traces):
    #Initialize arrays & variables to zero
    num_traces, num_points = traces.shape
    sumnum = np.zeros(num_points)
    sumden1 = np.zeros(num_points)
    sumden2 = np.zeros(num_points)

    #Mean of hypotheses
    h_mean = np.mean(hypotheses, dtype=np.float64)

    #Mean of all points in trace
    t_mean = np.mean(traces, axis=0, dtype=np.float64)

    #For each trace, do the following
    for t_idx in range(num_traces):
        h_diff = (hypotheses[t_idx] - h_mean)
        t_diff = traces[t_idx, :] - t_mean

        sumnum = sumnum + (h_diff * t_diff)
        sumden1 = sumden1 + h_diff * h_diff 
        sumden2 = sumden2 + t_diff * t_diff

    correlation = sumnum / np.sqrt(sumden1 * sumden2)

    return correlation

def corr_coef_vectorized(hypotheses, traces):
    h_mean = np.mean(hypotheses, axis=-1) # np.mean(hypotheses)
    t_mean = np.mean(traces, axis=-2) # np.mean(traces, axis=0)
    h_diff, t_diff = hypotheses - h_mean, traces - t_mean

    r_num = np.sum(h_diff[..., None] * t_diff, axis=-2) # np.sum(h_diff[:, None] * t_diff, axis=0)
    r_den = np.sqrt(np.sum(h_diff * h_diff, axis=-1) * np.sum(t_diff * t_diff, axis=-2)) # np.sqrt(np.sum(h_diff * h_diff, axis=0) * np.sum(t_diff * t_diff, axis=0))
    r = r_num / r_den
    r = np.clip(r, -1.0, 1.0)
    return r

s_boxes_b4 = [
    [0x0A, 0x06, 0x0B, 0x08, 0x04, 0x09, 0x08, 0x0C, 0x06, 0x0A, 0x05, 0x08, 0x0C, 0x07, 0x08, 0x04],
    [0x09, 0x01, 0x05, 0x05, 0x00, 0x0C, 0x02, 0x06, 0x07, 0x0F, 0x0B, 0x0B, 0x10, 0x04, 0x0E, 0x0A],
    [0x0D, 0x0E, 0x0E, 0x02, 0x03, 0x09, 0x03, 0x05, 0x03, 0x02, 0x02, 0x0E, 0x0D, 0x07, 0x0D, 0x0B],
    [0x02, 0x09, 0x08, 0x0B, 0x0D, 0x08, 0x01, 0x07, 0x0E, 0x07, 0x08, 0x05, 0x03, 0x08, 0x0F, 0x09],
    [0x0B, 0x03, 0x0F, 0x09, 0x00, 0x00, 0x0C, 0x00, 0x05, 0x0D, 0x01, 0x07, 0x10, 0x10, 0x04, 0x10],
    [0x0F, 0x0C, 0x01, 0x0F, 0x0E, 0x01, 0x06, 0x0C, 0x01, 0x04, 0x0F, 0x01, 0x02, 0x0F, 0x0A, 0x04],
    [0x06, 0x0E, 0x0D, 0x00, 0x07, 0x0E, 0x0C, 0x03, 0x0A, 0x02, 0x03, 0x10, 0x09, 0x02, 0x04, 0x0D],
    [0x0C, 0x00, 0x04, 0x01, 0x0F, 0x0B, 0x04, 0x00, 0x04, 0x10, 0x0C, 0x0F, 0x01, 0x05, 0x0C, 0x10],
    [0x0B, 0x00, 0x0F, 0x0A, 0x09, 0x0B, 0x09, 0x02, 0x05, 0x10, 0x01, 0x06, 0x07, 0x05, 0x07, 0x0E],
    [0x0D, 0x03, 0x0B, 0x0B, 0x08, 0x09, 0x08, 0x0C, 0x03, 0x0D, 0x05, 0x05, 0x08, 0x07, 0x08, 0x04],
    [0x0A, 0x02, 0x08, 0x04, 0x0F, 0x0B, 0x06, 0x04, 0x06, 0x0E, 0x08, 0x0C, 0x01, 0x05, 0x0A, 0x0C],
    [0x0D, 0x08, 0x0E, 0x08, 0x02, 0x05, 0x03, 0x0B, 0x03, 0x08, 0x02, 0x08, 0x0E, 0x0B, 0x0D, 0x05],
    [0x0D, 0x0F, 0x02, 0x05, 0x05, 0x0F, 0x09, 0x0B, 0x03, 0x01, 0x0E, 0x0B, 0x0B, 0x01, 0x07, 0x05],
    [0x0D, 0x00, 0x0A, 0x0A, 0x06, 0x07, 0x03, 0x0E, 0x03, 0x10, 0x06, 0x06, 0x0A, 0x09, 0x0D, 0x02],
    [0x00, 0x04, 0x07, 0x00, 0x09, 0x04, 0x0C, 0x00, 0x10, 0x0C, 0x09, 0x10, 0x07, 0x0C, 0x04, 0x10],
    [0x04, 0x0B, 0x06, 0x03, 0x0F, 0x06, 0x0C, 0x02, 0x0C, 0x05, 0x0A, 0x0D, 0x01, 0x0A, 0x04, 0x0E],
    [0x03, 0x0C, 0x01, 0x08, 0x08, 0x0F, 0x0D, 0x0F, 0x0D, 0x04, 0x0F, 0x08, 0x08, 0x01, 0x03, 0x01],
    [0x0B, 0x03, 0x02, 0x0C, 0x03, 0x08, 0x04, 0x02, 0x05, 0x0D, 0x0E, 0x04, 0x0D, 0x08, 0x0C, 0x0E]
]

s_boxes_b4 = [[s_ & 0x0F for s_ in box] for box in s_boxes_b4]

from ctypes import *

class aes_ctx(Structure):
    _fields_ = [
        ("RoundKey", c_uint8 * 176),
        ("Iv", c_uint8 * 16)
    ]

class ecrypt_ctx(Structure):
    _fields_ = [
        ("input", c_uint32 * 16)
    ]

class rng(Structure):
    pass

rng._fields_ = [
    ("indices", c_uint16 * 512),
    ("whitening", c_uint8 * 98),
    ("mode", c_int),
    ("gen_rand_uniform", CFUNCTYPE(c_uint8, POINTER(rng), POINTER(c_uint8))),
    ("copy", CFUNCTYPE(None, POINTER(rng), POINTER(rng))),
    ("next_elem", CFUNCTYPE(None, POINTER(rng)))
]

class rng_aes(Structure):
    _fields_ = [
        ("r", rng),
        ("ctx", aes_ctx),
        ("ctr", c_uint8 * 16),
        ("batch_idx", c_size_t)
    ]

class rng_cha(Structure):
    _fields_ = [
        ("r", rng),
        ("ctx", ecrypt_ctx),
        ("batch_idx", c_size_t)
    ]

lib = CDLL("../elisabeth/py_gen_rng.so")

def aes_random_4(seed: str):
    r = rng_aes()
    lib.rng_new_aes(byref(r), int(seed, 16).to_bytes(length=16, byteorder="little"), 1)
    return list(r.r.indices), list(r.r.whitening)
    
def aes_random_b4(seed: str):
    r = rng_aes()
    lib.rng_new_aes(byref(r), int(seed, 16).to_bytes(length=16, byteorder="little"), 0)
    return list(r.r.indices), list(r.r.whitening)
    
def chacha_random_4(seed: str):
    r = rng_cha()
    lib.rng_new_cha(byref(r), int(seed, 16).to_bytes(length=16, byteorder="little"), 1)
    return list(r.r.indices), list(r.r.whitening)
    
def chacha_random_b4(seed: str):
    r = rng_cha()
    lib.rng_new_cha(byref(r), int(seed, 16).to_bytes(length=16, byteorder="little"), 0)
    return list(r.r.indices), list(r.r.whitening)

def hypothesis_b4_rws_sboxes_location_hw(iv: str, key: List[int], round_idx: int, block_idx: int) -> int:
    indices, whitening = chacha_random_b4(iv)

    block = [(key[indices[i]] + whitening[i]) % 16 for i in range(BLOCK_WIDTH_B4 * round_idx, BLOCK_WIDTH_B4 * (round_idx + 1))]

    if block_idx != BLOCK_WIDTH_B4 - 1:
        if block_idx % 2 == 0:
            sbox_out = s_boxes_b4[block_idx][block[block_idx]]
        else:
            sbox_out = s_boxes_b4[block_idx][(block[block_idx] + block[block_idx - 1]) % 16]
        return HW[sbox_out]
    else:
        for i in range(3):
            block[2*i + 1] = (block[2*i + 1] + block[2*i]) % 16
        y = [s_boxes_b4[i][block[i]] for i in range(BLOCK_WIDTH_B4 - 1)]
        z = [(y[(2*i + 5*j - 1) % (BLOCK_WIDTH_B4 - 1)] + y[2*i + j]) % 16 for i in range(3) for j in range(2)]
        z = [s_boxes_b4[6 + i][(z[i] + block[(i + 2) % (BLOCK_WIDTH_B4 - 1)]) % 16] for i in range(BLOCK_WIDTH_B4 - 1)]
        t_0 = (z[0] + z[1] + z[2]) % 16
        t_0 = (t_0 + block[block_idx - 1]) % 16
        sbox_out = s_boxes_b4[12][t_0]

        return HW[(block[block_idx] + sbox_out) % 16]

def hypothesis_b4_rws_sboxes_location_hd(iv: str, key: List[int], round_idx: int, block_idx: int) -> int:
    indices, whitening = chacha_random_b4(iv)

    block = [(key[indices[i]] + whitening[i]) % 16 for i in range(BLOCK_WIDTH_B4 * round_idx, BLOCK_WIDTH_B4 * (round_idx + 1))]

    if block_idx != BLOCK_WIDTH_B4 - 1:
        if block_idx % 2 == 0:
            sbox_in = block[block_idx]
        else:
            sbox_in = (block[block_idx] + block[block_idx - 1]) % 16
        sbox_out = s_boxes_b4[block_idx][sbox_in]
        return HD[sbox_in][sbox_out]
    else:
        for i in range(3):
            block[2*i + 1] = (block[2*i + 1] + block[2*i]) % 16
        y = [s_boxes_b4[i][block[i]] for i in range(BLOCK_WIDTH_B4 - 1)]
        z = [(y[(2*i + 5*j - 1) % (BLOCK_WIDTH_B4 - 1)] + y[2*i + j]) % 16 for i in range(3) for j in range(2)]
        z = [s_boxes_b4[6 + i][(z[i] + block[(i + 2) % (BLOCK_WIDTH_B4 - 1)]) % 16] for i in range(BLOCK_WIDTH_B4 - 1)]
        t_0 = (z[0] + z[1] + z[2]) % 16
        t_0 = (t_0 + block[block_idx - 1]) % 16
        sbox_out = s_boxes_b4[12][t_0]

        return HD[block[block_idx] + sbox_out][(block[block_idx] + sbox_out) % 16]

def find_locations_in_time(seeds: np.ndarray, traces: np.ndarray, real_keys: np.ndarray, filename: str) -> np.ndarray:
    correlation_locations = []
    for round_idx in range(len(correlation_locations), KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4):
        corr_round = []
        for block_idx in range(BLOCK_WIDTH_B4):
            print(round_idx * BLOCK_WIDTH_B4 + block_idx, end="")
            hyps = np.array([hypothesis_b4_rws_sboxes_location_hw(iv, key, round_idx, block_idx) for i, key in enumerate(real_keys) for iv in seeds[i]])
            corr = corr_coef(hyps, traces.reshape((-1, traces.shape[2])))
            loc = np.argmax(corr)
            corr_round.append(list(range(loc - 5, loc + 5)))
            #plt.plot(corr)
            #plt.ylim([-0.5, 0.5])
            #plt.title(f"Round {round_idx}, Block {block_idx}, Location: {loc}")
            #plt.show()
        correlation_locations.append(corr_round)
    correlation_locations = np.array(correlation_locations)

    with open(filename, "wb") as w:
        pic.dump(correlation_locations, w)

    return correlation_locations

def load_data(traces_path: str, key_path: str, locations_path: str = "", max_traces: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traces_dict = sio.loadmat(traces_path, variable_names=[f"data_{i}" for i in range(max_traces)]) if max_traces is not None else sio.loadmat(traces_path)
    inputs_outputs, empty_traces = log_parser.parse(key_path)

    all_keys = [inputs_outputs[i][0][0] for i in range(1, len(inputs_outputs), 2)]
    unique_keys = []
    for i, k in enumerate(all_keys):
        if k not in [k for k, i in unique_keys]:
            unique_keys.append((k, i)) 

    real_keys = [np.array([int(c, 16) for c in key]) for key, i in unique_keys]
    
    traces = []
    seeds  = []
    for j in range(len(unique_keys) - 1):
        i1 = unique_keys[j][1]
        i2 = unique_keys[j + 1][1]

        traces.append(np.stack([traces_dict["data_" + str(i+1)][0, 0][4][:, 0] for i in range(i1, i2) if i+1 not in empty_traces]))
        seeds.append(np.stack([inputs_outputs[2 * i][0][0] for i in range(i1, i2) if i+1 not in empty_traces]))

    traces.append(np.stack([traces_dict["data_" + str(i+1)][0, 0][4][:, 0] for i in range(unique_keys[-1][1], len(all_keys)) if i+1 not in empty_traces]))
    seeds.append(np.stack([inputs_outputs[2 * i][0][0] for i in range(unique_keys[-1][1], len(all_keys)) if i+1 not in empty_traces]))

    assert len(traces) == len(seeds)
    assert all([t.shape[0] == s.shape[0] for t, s in zip(traces, seeds)])

    if locations_path:
        with open(locations_path, "rb") as r:
            correlation_locations = pic.load(r)
        return seeds, traces, real_keys, correlation_locations
    else:
        return seeds, traces, real_keys

def load_data_alternating_same_varying(traces_path: str, key_path: str, max_traces: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traces_dict = sio.loadmat(traces_path, variable_names=[f"data_{i}" for i in range(max_traces)]) if max_traces is not None else sio.loadmat(traces_path)
    inputs_outputs, empty_traces = log_parser.parse(key_path)

    all_keys = [inputs_outputs[i][0][0] for i in range(1, len(inputs_outputs), 2)]
    unique_keys = []
    for i, k in enumerate(all_keys):
        if k not in [k for k, i in unique_keys]:
            unique_keys.append((k, i)) 

    real_keys = [np.array([int(c, 16) for c in key]) for key, i in unique_keys]
    
    traces_varying = []
    traces_same = []
    seeds_varying  = []
    seeds_same = []
    for j in range(len(unique_keys) - 1):
        i1 = unique_keys[j][1]
        i2 = unique_keys[j + 1][1]

        traces_varying.append(np.stack([traces_dict["data_" + str(i+1)][0, 0][4][:, 0] for i in range(i1, i2) if i+1 not in empty_traces and i % 2 != 0]))
        traces_same.append(np.stack([traces_dict["data_" + str(i+1)][0, 0][4][:, 0] for i in range(i1, i2) if i+1 not in empty_traces and i % 2 == 0]))
        seeds_varying.append(np.stack([inputs_outputs[2 * i][0][0] for i in range(i1, i2) if i+1 not in empty_traces and i % 2 != 0]))
        seeds_same.append(np.stack([inputs_outputs[2 * i][0][0] for i in range(i1, i2) if i+1 not in empty_traces and i % 2 == 0]))

    traces_varying.append(np.stack([traces_dict["data_" + str(i+1)][0, 0][4][:, 0] for i in range(unique_keys[-1][1], len(all_keys)) if i+1 not in empty_traces and i % 2 != 0]))
    traces_same.append(np.stack([traces_dict["data_" + str(i+1)][0, 0][4][:, 0] for i in range(unique_keys[-1][1], len(all_keys)) if i+1 not in empty_traces and i % 2 == 0]))
    seeds_varying.append(np.stack([inputs_outputs[2 * i][0][0] for i in range(unique_keys[-1][1], len(all_keys)) if i+1 not in empty_traces and i % 2 != 0]))
    seeds_same.append(np.stack([inputs_outputs[2 * i][0][0] for i in range(unique_keys[-1][1], len(all_keys)) if i+1 not in empty_traces and i % 2 == 0]))

    assert len(traces_varying) == len(seeds_varying)
    assert all([t.shape[0] == s.shape[0] for t, s in zip(traces_varying, seeds_varying)])
    assert len(traces_same) == len(seeds_same)
    assert all([t.shape[0] == s.shape[0] for t, s in zip(traces_same, seeds_same)])
  
    return seeds_same, traces_same, seeds_varying, traces_varying, real_keys
    