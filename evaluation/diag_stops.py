import numpy as np
d = np.load("preprocessing/output/test_splits/Vta1a_test_split.npz")
X, v_end = d["X"], d["v_end"]
GYRO_W = -np.array([0.568, 0.831, 1.123])

tail = X[:, -10:, :]
acc = np.sqrt((tail[:, :, 0:3] ** 2).sum(-1)).mean(1)
gyr = np.abs(tail[:, :, 3:6] @ GYRO_W).mean(1)
stopped = v_end < 0.3
for name, f in (("acc", acc), ("gyr", gyr)):
    print(name, "stopped p50/p90:", np.percentile(f[stopped], [50, 90]),
          "moving p10/p50:", np.percentile(f[~stopped], [10, 50]))
print("fraction of windows that are stops:", stopped.mean())
