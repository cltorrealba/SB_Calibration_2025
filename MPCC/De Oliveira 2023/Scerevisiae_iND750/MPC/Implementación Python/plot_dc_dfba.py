# plot_panel_2x2.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Constantes (mismas del modelo) ===
conv_glu = 0.180156   # g/mmol
conv_eth = 0.04607    # g/mmol
Osat     = 3.0e-4     # mol/L  (100% DO)

HERE = Path(__file__).parent

def load_csv(name):
    p = HERE / name
    if not p.exists():
        raise FileNotFoundError(f"Falta el archivo: {name}")
    return np.loadtxt(p, delimiter=",")

def infer_edges_from_collocation(tsn, NCP=3):
    """Dado tsn (incluye el punto inicial), devuelve bordes de elementos finitos."""
    L = len(tsn)
    if (L-1) % NCP != 0:
        raise ValueError("tsn no es compatible con NCP=3 (Radau).")
    NFE = (L-1) // NCP
    edges = [tsn[0]]
    for i in range(NFE):
        idx_last = 1 + i*NCP + (NCP-1)
        edges.append(tsn[idx_last])
    return np.array(edges), NFE

def stepify(edges, values_per_fe):
    """Convierte valores por elemento finito a series 'step' en el tiempo."""
    edges = np.asarray(edges).ravel()
    NFE = len(edges) - 1
    vals = np.asarray(values_per_fe)
    if vals.ndim == 1:
        vals = vals[None, :]
    # Asegurar forma (K, NFE)
    if vals.shape[-1] != NFE:
        if vals.shape[0] == NFE:
            vals = vals.T
        else:
            raise ValueError("values_per_fe debe tener longitud NFE.")
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

def main(tsn_file="tsn_py.csv", xk_file="xk_py.csv", uk_file="uk_py.csv",
         save_path="dcdfba_panel_2x2.png"):
    # Cargar resultados
    tsn = load_csv(tsn_file)            # 1 + NFE*NCP
    X   = load_csv(xk_file)             # (1 + NFE*NCP) x 4 : [X(g), G(mmol), E(mmol), V(L)]
    uk  = load_csv(uk_file)             # 2 x NFE : fila0=DO [mol/L], fila1=F [L/h]

    # Inferir bordes de elementos
    edges, NFE = infer_edges_from_collocation(tsn, NCP=3)

    # Asegurar forma (2, NFE) en uk
    uk = np.asarray(uk)
    if uk.ndim != 2:
        raise ValueError("uk_py.csv debe ser 2D (2 x NFE).")
    if uk.shape[0] != 2 and uk.shape[1] == 2:
        uk = uk.T
    if uk.shape[0] != 2:
        raise ValueError("uk_py.csv debe tener dos filas: [DO; F].")
    if uk.shape[1] != NFE:
        raise ValueError("uk_py.csv columnas deben coincidir con NFE inferido.")

    # Estados y conversiones
    Xg   = X[:, 0]               # Biomasa total [g]
    Gmm  = X[:, 1]               # Glucosa total [mmol]
    Emm  = X[:, 2]               # Etanol total [mmol]
    V    = X[:, 3]               # Volumen [L]

    # Concentraciones (g/L)
    G_gL = (Gmm / V) * conv_glu
    E_gL = (Emm / V) * conv_eth
    X_gL = Xg / V

    # Controles por elemento finito
    DO_molL = uk[0, :]                  # mol/L (del solver)
    F_Lh    = uk[1, :]                  # L/h   (del solver)

    # Series escalón en tiempo
    t_ctrl, (DO_step, F_step) = stepify(edges, np.vstack([DO_molL, F_Lh]))

    # Conversiones pedidas
    F_Lmin   = F_step                             # L/h
    DO_pct   = (DO_step / Osat) * 100.0           # % de saturación

    # === Figura 2x2 ===
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # (0,0) Volumen con límite 1.2 L
    ax = axs[0, 0]
    ax.plot(tsn, V, label="Volume (L)")
    ax.axhline(1.2, linestyle=":", label="Limit 1.2 L")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Volume (L)")
    ax.set_title("Volume")
    ax.legend(loc="best")

    # (0,1) g/L (glucosa, etanol) + biomasa g/L en eje derecho
    ax = axs[0, 1]
    l1, = ax.plot(tsn, G_gL, label="Glucose (g/L)", color="red")
    l2, = ax.plot(tsn, E_gL, label="Ethanol (g/L)", color="green")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (g/L)")
    ax2 = ax.twinx()
    l3, = ax2.plot(tsn, X_gL, label="Biomass (g/L)", color="blue")
    ax2.set_ylabel("Biomass (g/L)")
    ax.set_title("Concentrations")
    # Leyenda combinada
    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="best")

    # (1,0) F en L/min (escalón)
    ax = axs[1, 0]
    ax.step(t_ctrl, F_Lmin, where="post", label="Glucose feed F (L/min)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("F (L/h)")
    ax.set_title("Glucose feed")
    ax.legend(loc="best")

    # (1,1) DO en % (escalón)
    ax = axs[1, 1]
    ax.step(t_ctrl, DO_pct, where="post", label="Dissolved Oxygen (% sat.)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("DO (%)")
    ax.set_title("Dissolved Oxygen")
    ax.legend(loc="best")

    fig.suptitle("Dynamic optimization simulation (iND750, DC dFBA)", fontsize=12)
    fig.savefig(HERE / save_path, dpi=200)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
