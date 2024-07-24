from typing import List
import pickle as pic

import numpy as np
#import cupy as np
import matplotlib.pyplot as plt
from utils import *

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
    correlation_locations = [[[0] * 10] * BLOCK_WIDTH_B4] * (KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 - 2)
    for round_idx in range(len(correlation_locations), KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4):
        corr_round = []
        for block_idx in range(BLOCK_WIDTH_B4):
            hyps = np.array([hypothesis_b4_rws_sboxes_location_hw(iv, key, round_idx, block_idx) for i, key in enumerate(real_keys) for iv in seeds[i]])
            corr = corr_coef(hyps, traces.reshape((-1, traces.shape[2])))
            loc = np.argmax(corr)
            corr_round.append(list(range(loc - 5, loc + 5)))
            plt.plot(corr)
            plt.ylim([-0.5, 0.5])
            plt.title(f"Round {round_idx}, Block {block_idx}, Location: {loc}")
            plt.show()
        correlation_locations.append(corr_round)
    correlation_locations = np.array(correlation_locations)

    with open(filename, "wb") as w:
        pic.dump(correlation_locations, w)

    return correlation_locations
