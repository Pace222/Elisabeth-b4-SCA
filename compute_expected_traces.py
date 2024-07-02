import numpy as np
from math import comb

def solve(N: int, n: int):
    P = np.empty((N + 1, N + 1), dtype=np.float64)
    for i in range(N + 1):
        for j in range(N + 1):
            if i <= j <= i + n:
                P[i, j] = comb(N - i, j - i) * comb(i, n - (j - i)) / comb(N, n)
            else:
                P[i, j] = 0

    Q = P[:N, :N]
    NN = np.linalg.inv(np.eye(N) - Q)
    c = np.ones(N)
    t = np.dot(NN, c)

    return t[0]


if __name__ == '__main__':
    E = solve(N=512, n=98)
    print(E)
    