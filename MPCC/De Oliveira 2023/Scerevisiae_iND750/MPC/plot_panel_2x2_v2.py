"""
Plot panel 2x2 (Volume, Concentrations, F, DO) + glycerol concentration (g/L).
Requires: tsn_py.csv, xk_py.csv, uk_py.csv; optional: v_py.csv for glycerol plot.

Set GLY_EX_IDX (0-based) to the glycerol exchange reaction index (SBML order).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Constants (same as model) ===
conv_glu = 0.180156   # g/mmol
conv_eth = 0.04607    # g/mmol
Osat     = 3.0e-4     # mol/L  (100% DO)

# >>>>>>>>>>>>  GLYCEROL EXCHANGE RXN INDEX (0-based)  <<<<<<<<<<<<
GLY_EX_IDX = 401  # R_EX_glyc_e (Glycerol exchange), 0-based

HERE = Path(__file__).parent

def load_csv(name, required=True):
    p = HERE / name
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {name}")
        return None
    return np.loadtxt(p, delimiter=",")

def infer_edges_from_collocation(tsn, NCP=3):
    L = len(tsn)
    if (L-1) % NCP != 0:
        raise ValueError("tsn not compatible with NCP=3 (Radau).")
    NFE = (L-1) // NCP
    edges = [tsn[0]]
    for i in range(NFE):
        idx_last = 1 + i*NCP + (NCP-1)
        edges.append(tsn[idx_last])
    return np.array(edges), NFE

def stepify(edges, values_per_fe):
    edges = np.asarray(edges).ravel()
    NFE = len(edges) - 1
    vals = np.asarray(values_per_fe)
    if vals.ndim == 1:
        vals = vals[None, :]
    if vals.shape[-1] != NFE:
        if vals.shape[0] == NFE:
            vals = vals.T
        else:
            raise ValueError("values_per_fe length must be NFE.")
    t = []
    Y = [[] for _ in range(vals.shape[0])]
    for i in range(NFE):
        a, b = edges[i], edges[i+1]
        if i == 0:
            t.append(a)
            for k in range(vals.shape[0]):
                Y[k].append(vals[k, i])
        t.append(b)
        for k in range(vals.shape[0]):
            Y[k].append(vals[k, i])
    return np.array(t), [np.array(y) for y in Y]

def main(tsn_file="tsn_py.csv", xk_file="xk_py.csv", uk_file="uk_py.csv", v_file="v_py.csv",
         save_panel="dcdfba_panel_2x2_v2.png", save_gly="glycerol_concentration.png"):
    tsn = load_csv(tsn_file)
    X   = load_csv(xk_file)
    uk  = load_csv(uk_file)
    vfe = load_csv(v_file, required=False)  # (NV, NFE)

    edges, NFE = infer_edges_from_collocation(tsn, NCP=3)

    # Ensure uk shape = (2,NFE)
    uk = np.asarray(uk)
    if uk.ndim != 2:
        raise ValueError("uk_py.csv must be 2D (2 x NFE).")
    if uk.shape[0] != 2 and uk.shape[1] == 2:
        uk = uk.T
    if uk.shape[0] != 2:
        raise ValueError("uk_py.csv must have shape (2, NFE).")
    if uk.shape[1] != NFE:
        raise ValueError("uk_py.csv columns must equal inferred NFE.")

    # States and concentrations
    Xg  = X[:,0]  # g
    Gmm = X[:,1]  # mmol
    Emm = X[:,2]  # mmol
    V   = X[:,3]  # L

    G_gL = (Gmm / V) * conv_glu
    E_gL = (Emm / V) * conv_eth
    X_gL = Xg / V

    # Controls
    DO_molL = uk[0,:]
    F_Lh    = uk[1,:]
    t_ctrl, (DO_step, F_step) = stepify(edges, np.vstack([DO_molL, F_Lh]))
    F_Lmin = F_step / 60.0
    DO_pct = (DO_step / Osat) * 100.0

    # === 2x2 panel ===
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # (0,0) Volume with 1.2 L limit
    ax = axs[0,0]
    ax.plot(tsn, V, label="Volume (L)")
    ax.axhline(1.2, linestyle=":", label="Limit 1.2 L")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Volume (L)")
    ax.set_title("Volume")
    ax.legend(loc="best")

    # (0,1) Concentrations (g/L) left axis, biomass g/L right axis
    ax = axs[0,1]
    l1, = ax.plot(tsn, G_gL, label="Glucose (g/L)")
    l2, = ax.plot(tsn, E_gL, label="Ethanol (g/L)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (g/L)")
    ax2 = ax.twinx()
    l3, = ax2.plot(tsn, X_gL, label="Biomass (g/L)")
    ax2.set_ylabel("Biomass (g/L)")
    ax.set_title("Concentrations")
    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="best")

    # (1,0) F in L/min (step)
    ax = axs[1,0]
    ax.step(t_ctrl, F_Lmin, where="post", label="Glucose feed F (L/min)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("F (L/min)")
    ax.set_title("Glucose feed")
    ax.legend(loc="best")

    # (1,1) DO in % (step)
    ax = axs[1,1]
    ax.step(t_ctrl, DO_pct, where="post", label="Dissolved Oxygen (% sat.)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("DO (%)")
    ax.set_title("Dissolved Oxygen")
    ax.legend(loc="best")

    fig.suptitle("Dynamic optimization simulation (iND750, DC dFBA) — v2", fontsize=12)
    fig.savefig(HERE / save_panel, dpi=200)
    print(f"Saved {save_panel}")

    # === Extra: glycerol concentration (g/L) from integrated production ===
    if vfe is not None and GLY_EX_IDX is not None:
        MW_GLY = 92.09382  # g/mol
        vfe = np.asarray(vfe)
        # expect (NV, NFE); if (NFE, NV), transpose
        if vfe.ndim == 2 and vfe.shape[0] < vfe.shape[1]:
            vfe = vfe.T
        if GLY_EX_IDX < 0 or GLY_EX_IDX >= vfe.shape[0]:
            raise IndexError(f"GLY_EX_IDX={GLY_EX_IDX} out of range for v_py.csv with NV={vfe.shape[0]}")
        v_gly = vfe[GLY_EX_IDX, :]  # mmol/gDW·h per FE (piecewise-constant)

        # Build rate r(tk) = v_gly(fe_k) * Xg(tk) at collocation nodes
        NCP = 3
        N_nodes = len(tsn)
        r_nodes = np.zeros(N_nodes)  # mmol/h
        for k in range(N_nodes):
            fe_k = 0 if k == 0 else ( (k-1) // NCP )
            fe_k = min(fe_k, len(v_gly)-1)
            r_nodes[k] = v_gly[fe_k] * Xg[k]

        # Cumulative trapezoidal integration over tsn
        n_mmol = np.zeros(N_nodes)  # accumulated glycerol (mmol)
        for k in range(N_nodes-1):
            dt = tsn[k+1] - tsn[k]    # hours
            n_mmol[k+1] = n_mmol[k] + 0.5*(r_nodes[k] + r_nodes[k+1]) * dt

        # Convert to g/L: (mmol * g/mol / 1000) / V
        gly_g_per_L = (n_mmol * MW_GLY / 1000.0) / V

        fig2, axg = plt.subplots(figsize=(8,4))
        axg.plot(tsn, gly_g_per_L, label="Glycerol (g/L)")
        axg.set_xlabel("Time (h)")
        axg.set_ylabel("Concentration (g/L)")
        axg.set_title("Glycerol concentration")
        axg.legend(loc="best")
        fig2.savefig(HERE / save_gly, dpi=200)
        print(f"Saved {save_gly}")
    else:
        print("Skipping glycerol plot: v_py.csv not found or GLY_EX_IDX not set.")

if __name__ == "__main__":
    main()
