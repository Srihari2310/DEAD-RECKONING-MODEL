import pandas as pd
import numpy as np
import sys

trip = sys.argv[1] if len(sys.argv) > 1 else "Vw3"
path = f"preprocessing/output/03_filtered/S-{trip}_filtered.csv"

df = pd.read_csv(path)
df.columns = df.columns.str.strip()

cols = ["accel_forward_filt", "accel_lateral_filt", "accel_vertical_filt"]
# adjust names above if yours differ -- check actual columns first:
print("Columns available:", [c for c in df.columns if "accel" in c.lower()])
print()

for c in cols:
    if c not in df.columns:
        continue
    s = df[c]
    print(f"{c}:")
    print(f"  mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.3f}  max={s.max():.3f}")
    # physically implausible if regularly beyond +-15 m/s^2 for a normal car
    extreme = (s.abs() > 15).sum()
    print(f"  extreme (|value|>15 m/s^2): {extreme} rows ({100*extreme/len(s):.2f}%)")
    print()