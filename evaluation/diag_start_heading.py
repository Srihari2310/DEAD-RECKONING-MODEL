import numpy as np

d = np.load("preprocessing/output/06_windows/Vta1a_windows.npz")
print("keys:", d.files)   # if 'heading_start' isn't listed, tell me the real name
X = d["X"]
hs = d["heading_start"]
hs_deg = np.degrees(hs) if np.abs(hs).max() <= 7 else hs.astype(float)
hs_unwrapped = np.degrees(np.unwrap(np.radians(hs_deg)))

GYRO_W = -np.array([0.568, 0.831, 1.123])
gyro_rate = X[:, :, 3:6] @ GYRO_W                       # rad/s
step_deg = np.degrees(gyro_rate[:, :10].sum(axis=1) * 0.1)
gyro_cum = np.concatenate([[0.0], np.cumsum(step_deg)])[:len(hs_deg)]

print("win | true_cum_deg | gyro_cum_deg | gyro_minus_true")
for w in range(0, 61):
    t = hs_unwrapped[w] - hs_unwrapped[0]
    print(f"{w:3d} | {t:11.1f} | {gyro_cum[w]:11.1f} | {gyro_cum[w]-t:11.1f}")
