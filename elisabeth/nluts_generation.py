import hashlib
import random

from typing import List

def gen_nluts_b4_precompute_shares() -> List[List[int]]:
    nluts_orig = gen_nluts_b4()
    nluts_shares = [[[[] for share in range(16)] for j in range(16)] for i in range(18)]
    for i in range(18):
        for j in range(16):
            for share in range(16):
                nluts_shares[i][share][(j - share) % 16] = (nluts_orig[i][j] - share) % 16

    return nluts_shares

def gen_nluts_b4_packed() -> List[List[int]]:
    nluts_orig = gen_nluts_b4()
    nluts_packed = []
    for i in range(18):
        nlut = []
        for j in range(0, 16, 4):
            pack = 0
            for k in range(4):
                pack |= nluts_orig[i][j + k] << 8*(3 - k)
            nlut.append(pack)
        nluts_packed.append(nlut)

    return nluts_packed

def gen_nluts_b4() -> List[List[int]]:
    str_seed = "Welcome to Gabriel"
    seed = int(hashlib.sha256(str_seed.encode("UTF-8")).hexdigest(), 16)
    random.seed(seed)
    
    nluts = []
    for i in range(18):
        nlut = []
        for j in range(8):
            nlut.append(random.randrange(16))
        for j in range(8):
            nlut.append(16 - nlut[j])
        nluts.append(nlut)
    
    return nluts

def gen_nluts_4() -> List[List[int]]:
    str_seed = "Welcome to Elisabeth, heir of FiLIP!"
    seed = hashlib.sha256(str_seed.encode("UTF-8")).digest()

    nluts = []
    for i in range(8):
        nlut = []
        for j in range(4):
            nlut.append(seed[4*i + j] >> 4)
            nlut.append(seed[4*i + j] & 0xF)
        for j in range(8):
            nlut.append(16 - nlut[j])
        nluts.append(nlut)

    return nluts

def print_nluts(nluts: List[List[List[int]]]):
    for nlut in nluts:
        print("{")
        for n in nlut:
            print("  {", end = "")
            print(", ".join(["0x{:02X}".format(nn) for nn in n]), end = "")
            print("},")
        print("},")

if __name__ == '__main__':
    print_nluts(gen_nluts_b4_precompute_shares())