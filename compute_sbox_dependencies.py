import numpy as np

def solve():
    res = np.zeros((18, 7), dtype=bool)
    x = np.eye(7, dtype=bool)
    for i in range(3):
        x[2*i+1] |= x[2*i]
    y = np.zeros((6, 7), dtype=bool)
    for i in range(6):
        res[i] |= x[i]
        y[i] |= res[i]
    z = np.zeros((6, 7), dtype=bool)
    for i in range(3):
        z[2*i] |= y[(2*i + 5) % 6] | y[2*i]
        z[2*i+1] |= y[(2*i + 4) % 6] | y[2*i + 1]
    for i in range(6):
        z[i] |= x[(i+2) % 6]
        res[i+6] |= z[i]
        z[i] |= res[i+6]
    t = np.zeros((6, 7), dtype=bool)
    for i in range(2):
        t[3*i] |= z[3*i] | z[3*i+1] | z[3*i+2]
        t[3*i+1] |= z[3*i+1] | z[(3*i+3) % 6]
        t[3*i+2] |= z[3*i+2] | z[(3*i+3) % 6] | y[3*i]
    t[0] |= x[5]
    t[1] |= x[4]
    t[2] |= x[3]
    t[3] |= x[1]
    t[4] |= x[0]
    t[5] |= x[2]
    r = x[6]
    for i in range(6):
        res[i+12] = t[i]
        r |= res[i+12]
    return res

if __name__ == '__main__':
    results = solve()
    for i, res in enumerate(results):
        print(f"S-box {i+1} depends on {np.sum(res)} ({res}) inputs")