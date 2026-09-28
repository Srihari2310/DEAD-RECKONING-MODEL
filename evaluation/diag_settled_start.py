import numpy as np

d = np.load("preprocessing/output/06_windows/Vta1a_windows.npz")
hs, v = d["heading_start"], d["v_start"]
X = d["X"]
hs = np.degrees(hs) if np.abs(hs).max() <= 7 else hs.astype(float)
GYRO_W = -np.array([0.568, 0.831, 1.123])
step = np.degrees((X[:, :, 3:6] @ GYRO_W)[:, :10].sum(axis=1) * 0.1)
wrap = lambda a: (a + 180) % 360 - 180

def settled_index(v, hs, min_speed=3.0, run=5, max_change=5.0):
    """First window where the previous `run` windows were all fast and steady."""
    for i in range(run, len(v)):
        seg_v = v[i - run:i + 1]
        seg_h = np.abs(wrap(np.diff(hs[i - run:i + 1])))
        if (seg_v >= min_speed).all() and (seg_h < max_change).all():
            return i
    return 0

f = settled_index(v, hs)
print("settled start window:", f)
for start in (24, f):
    est = hs[start] + np.concatenate([[0], np.cumsum(step[start:])])
    print(f"--- start {start} ---")
    for w in (100, 500, 1500, 1706, 2102, 2549):
        print(f"window {w}: err {wrap(est[w - start] - hs[w]):7.1f} deg")
