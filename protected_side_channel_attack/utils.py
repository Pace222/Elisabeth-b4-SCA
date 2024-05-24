import numpy as np
import plotly.express as px

KEYROUND_WIDTH_4 = 60
KEYROUND_WIDTH_B4 = 98

KEY_WIDTH_4 = 256
KEY_WIDTH_B4 = 512

BLOCK_WIDTH_4 = 5
BLOCK_WIDTH_B4 = 7

NR_SHARES = 2
NR_PERMS = 2 + 14*8

EARLIEST_ROUND = 0
LATEST_ROUND = 1

KEY_ALPHABET = list(range(16))

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

def plot(line: np.ndarray, name: str, title: str, xaxis_title: str, yaxis_title: str, is_t_test: bool = False, showlegend: bool= True, xaxis: dict = {}, yaxis_range: list = []):
    fig = px.line(line)
    fig.for_each_trace(lambda t: t.update(name=name))
    if is_t_test:
        fig.add_hline(4.5, line_color="red")
        fig.add_hline(-4.5, line_color="red")

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=showlegend
    )
    if len(xaxis) > 0:
        fig.update_layout(xaxis=xaxis)
    if len(yaxis_range) > 0:
        fig.update_layout(yaxis_range=yaxis_range)

    config = {'scrollZoom': True}
    fig.show(config=config)