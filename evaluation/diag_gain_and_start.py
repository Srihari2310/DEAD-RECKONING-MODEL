import numpy as np

d = np.load("preprocessing/output/06_windows/Vta1a_windows.npz")
X, hs, v = d["X"], d["heading_start"], d["v_start"]
hs = np.degrees(hs) if np.abs(hs).max() <= 7 else hs.astype(float)
GYRO_W = -np.array([0.568, 0.831, 1.123])
gyro_rate = X[:, :, 3:6] @ GYRO_W
step = np.degrees(gyro_rate[:, :10].sum(axis=1) * 0.1)          # gyro deg per window
wrap = lambda a: (a + 180) % 360 - 180

# 1) gain: true heading change vs gyro change, only at real speed, skipping glitches
dt_true = wrap(hs[1:] - hs[:-1])
ok = (v[:-1] >= 3) & (v[1:] >= 3) & (np.abs(dt_true) < 60)
g, t = step[:-1][ok], dt_true[ok]
gain = (g @ t) / (g @ g)
print(f"windows used: {ok.sum()}  corr: {np.corrcoef(g, t)[0,1]:.3f}  gain(true/gyro): {gain:.3f}")

# 2) heading tracking from first fast window, plain and gain-corrected
f = int(np.argmax(v >= 3.0))
print("first_fast window:", f)
for label, k in (("plain", 1.0), ("gain-corrected", gain)):
    est = hs[f] + np.concatenate([[0], np.cumsum(step[f:] * k)])
    print(f"--- {label} ---")
    for w in (50, 500, 1500, 1706, 2102, 2549):
        print(f"window {w}: err {wrap(est[w - f] - hs[w]):7.1f} deg")
