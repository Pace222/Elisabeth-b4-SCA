import numpy as np

from ctypes import *

# Constant definitions
KEYROUND_WIDTH_4 = 60
KEYROUND_WIDTH_B4 = 98

KEY_WIDTH_4 = 256
KEY_WIDTH_B4 = 512

BLOCK_WIDTH_4 = 5
BLOCK_WIDTH_B4 = 7

NR_SHARES = 2
NR_PERMS = 2 + 14*8

EARLIEST_ROUND = 0
LATEST_ROUND = 14

KEY_ALPHABET = list(range(16))

HW = [bin(n).count("1") for n in range(0, 128)]
HD = [[HW[n1 ^ n2] for n2 in range(0, 128)] for n1 in range(0, 128)]

def corr_coef(hypotheses: np.ndarray, traces: np.ndarray) -> np.ndarray:
    """Computes the Pearson correlation coefficient between the consumption model and the observed traces.
    This function is rather slow because it computes the coefficients iteratively on all traces.

    Args:
        hypotheses (_type_): Model values according to a certain key hypothesis
        traces (_type_): Observed traces

    Returns:
        _type_: Correlation coefficient at every time point
    """
    num_traces, num_points = traces.shape
    sumnum = np.zeros(num_points)
    sumden1 = np.zeros(num_points)
    sumden2 = np.zeros(num_points)

    # Mean of hypotheses
    h_mean = np.mean(hypotheses, dtype=np.float64)

    # Mean of all points in trace
    t_mean = np.mean(traces, axis=0, dtype=np.float64)

    # For each trace, do the following
    for t_idx in range(num_traces):
        h_diff = (hypotheses[t_idx] - h_mean)
        t_diff = traces[t_idx, :] - t_mean

        sumnum = sumnum + (h_diff * t_diff)
        sumden1 = sumden1 + h_diff * h_diff 
        sumden2 = sumden2 + t_diff * t_diff

    correlation = sumnum / np.sqrt(sumden1 * sumden2)

    return correlation

def corr_coef_vectorized(hypotheses: np.ndarray, traces: np.ndarray) -> np.ndarray:
    """Computes the Pearson correlation coefficient between the consumption model and the observed traces.
    This function is fast because it computes the coefficients in a vectorized way

    Args:
        hypotheses (_type_): Model values according to a certain key hypothesis
        traces (_type_): Observed traces

    Returns:
        _type_: Correlation coefficient at every time point
    """
    h_mean = np.mean(hypotheses, axis=-1)
    t_mean = np.mean(traces, axis=-2)
    h_diff, t_diff = hypotheses - h_mean, traces - t_mean

    r_num = np.sum(h_diff[..., None] * t_diff, axis=-2)
    r_den = np.sqrt(np.sum(h_diff * h_diff, axis=-1) * np.sum(t_diff * t_diff, axis=-2))
    r = r_num / r_den
    r = np.clip(r, -1.0, 1.0)
    return r


"""
C shared library API for Python
"""

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

lib = CDLL("../../elisabeth/py_gen_rng.so")

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
