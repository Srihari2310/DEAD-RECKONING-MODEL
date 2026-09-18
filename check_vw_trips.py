import numpy as np
from pathlib import Path

trips = ['Vw2','Vw3','Vw4','Vw5','Vw6','Vw7','Vw8','Vw9','Vw10','Vw11','Vw12','Vw14a','Vw14b','Vw14c','Vw16a','Vw16b','Vw17']
for t in trips:
    p = Path(f'preprocessing/output/06_windows/{t}_windows.npz')
    if not p.exists():
        print(f'{t}: MISSING (not processed or failed)')
        continue
    d = np.load(p)
    yl = np.linalg.norm(d['y_local'], axis=1).mean()
    yg = np.linalg.norm(d['y_global'], axis=1).mean()
    n = len(d['X'])
    print(f'{t}: windows={n}, |y_local|={yl:.2f}m, |y_global|={yg:.2f}m, match={abs(yl-yg)<0.5}')
