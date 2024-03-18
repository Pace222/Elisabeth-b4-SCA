from typing import Tuple, List, Generator

from Crypto.Cipher import AES
from Crypto.Cipher import ChaCha20


def rand_uniform_chacha(cipher: ChaCha20.ChaCha20Cipher) -> Generator[int, None, None]:
    while True:
        encrypted = cipher.encrypt(b"\x00" * 64)
        for i in range(64):
            yield encrypted[i]

def rand_uniform_aes(cipher) -> Generator[int, None, None]:
    counter = 0
    while True:
        encrypted = cipher.encrypt(counter.to_bytes(length=16, byteorder="little"))
        for i in range(16):
            yield encrypted[i]
        counter += 1

def random_uniform_n_lsb(rand_generator: Generator[int, None, None], n: int) -> int:
    rnd = 0
    
    for i in range(4):
        rnd |= next(rand_generator) << (i * 8)

    return rnd >> (32 - n)

def gen_range(rand_generator: Generator[int, None, None], min: int, max: int) -> int:
    if min > max:
        return -1
    bit_len = (max - min).bit_length() - 1
    a = min + random_uniform_n_lsb(rand_generator, bit_len)

    while (a >= max):
        a = min + random_uniform_n_lsb(rand_generator, bit_len)
    
    return a


def precompute_prng(seed: int, algo: str, mode: bool) -> Tuple[List[int], List[int]]:
    KEY_WIDTH = 256 if mode else 512
    KEYROUND_WIDTH = 60 if mode else 98

    if algo == "AES":
        cipher = AES.new(seed.to_bytes(length=16, byteorder="little"), AES.MODE_ECB)
        rand_uniform = rand_uniform_aes(cipher)
    elif algo == "ChaCha":
        cipher = ChaCha20.new(key=seed.to_bytes(length=32, byteorder="little"), nonce=b"\x00"*8)
        rand_uniform = rand_uniform_chacha(cipher)
    else:
        raise ValueError()

    indices = list(range(KEY_WIDTH))
    for i in range(KEYROUND_WIDTH):
        j = gen_range(rand_uniform, i, KEY_WIDTH)
        tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp

    whitening = [next(rand_uniform) % 16 for _ in range(KEYROUND_WIDTH)]

    return indices, whitening

def aes_random_4(seed: int) -> Tuple[List[int], List[int]]:
    return precompute_prng(seed, "AES", True)
    
def aes_random_b4(seed: int) -> Tuple[List[int], List[int]]:
    return precompute_prng(seed, "AES", False)

def chacha_random_4(seed: int) -> Tuple[List[int], List[int]]:
    return precompute_prng(seed, "ChaCha", True)

def chacha_random_b4(seed: int) -> Tuple[List[int], List[int]]:
    return precompute_prng(seed, "ChaCha", False)