import numpy as np

d = np.load("preprocessing/output/06_windows/Vta1a_windows.npz")
hs, v, X = d["heading_start"], d["v_start"], d["X"]
hs = np.degrees(hs) if np.abs(hs).max() <= 7 else hs.astype(float)
GYRO_W = -np.array([0.568, 0.831, 1.123])
step = np.degrees((X[:, :, 3:6] @ GYRO_W)[:, :10].sum(axis=1) * 0.1)
wrap = lambda a: (a + 180) % 360 - 180

START = 36
est = hs[START] + np.concatenate([[0], np.cumsum(step[START:])])
err = np.array([wrap(est[w - START] - hs[w]) for w in range(START, len(hs))])
w_idx = np.arange(START, len(hs))

# 1) where does the error change fastest? (20-window blocks)
print("Biggest error changes per 20-window block:")
blocks = []
for a in range(START, len(hs) - 20, 20):
    delta = err[a + 20 - START] - err[a - START]
    blocks.append((abs(delta), a, delta, v[a:a + 20].mean(), np.abs(step[a:a + 20]).sum()))
for _, a, delta, vm, rot in sorted(blocks, reverse=True)[:10]:
    print(f"windows {a}-{a+20}: err change {delta:7.1f} deg | mean speed {vm:5.1f} m/s | total |gyro turn| {rot:6.1f} deg")

# 2) gain by speed bin (true heading change vs gyro change)
dt_true = wrap(hs[1:] - hs[:-1])
print("\nGain by speed bin (true/gyro), turning windows only (|gyro step|>2 deg):")
for lo, hi in ((3, 6), (6, 10), (10, 15), (15, 40)):
    m = (v[:-1] >= lo) & (v[:-1] < hi) & (np.abs(dt_true) < 60) & (np.abs(step[:-1]) > 2)
    if m.sum() > 20:
        g, t = step[:-1][m], dt_true[m]
        print(f"speed {lo:2d}-{hi:2d} m/s: n={m.sum():4d}  gain={(g @ t)/(g @ g):.3f}  corr={np.corrcoef(g, t)[0,1]:.3f}")
