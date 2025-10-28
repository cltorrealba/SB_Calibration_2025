import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
ts = np.loadtxt(HERE/"tsn_py.csv", delimiter=",")
X  = np.loadtxt(HERE/"xk_py.csv", delimiter=",")

# X[:,0]=biomass(g), X[:,1]=G(mmol), X[:,2]=E(mmol), X[:,3]=V(L)
fig, ax = plt.subplots()
ax.plot(ts, X[:,0], label="Biomass (g)")
ax.plot(ts, X[:,1]/X[:,3], label="Glucose (mmol/L)")
ax.plot(ts, X[:,2]/X[:,3], label="Ethanol (mmol/L)")
ax.set_xlabel("Time (h)")
ax.set_ylabel("States")
ax.legend()
fig.tight_layout()
fig.savefig(HERE/"fig8_like.png", dpi=180)
print("Saved fig8_like.png")