import os
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sklearn

# ---------------- Config (ajusta si difiere) ----------------
DENS_SHEET = "Manual Densidades"
DENS_DATE_COL = "medicion_fecha"      # col fecha-hora en hoja densidad
DENS_VALUE_CANDIDATES = ["densidad", "density"]  # candidatos
TEMPS_DIR = "Datos Experimentales"    # misma carpeta usada para Data <ID>.xlsx
POLY_DEGREE = 3                       # grado polinomio (puede pasarse por CLI)
MIN_POINTS_PER_ASSAY = 4              # mínimo para incluir ensayo
MAX_TIME_DIFF_H = 2.0                 # tolerancia para match nearest si no interpolamos
# ------------------------------------------------------------

def _find_density_col(df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in df.columns]
    low = {c.lower(): c for c in cols}
    for cand in DENS_VALUE_CANDIDATES:
        for k, orig in low.items():
            if cand in k:
                return orig
    return None

def _load_density_table(assay_code: str,
                        sb2id: Dict[str, int]) -> Optional[pd.DataFrame]:
    ens_id = sb2id.get(assay_code)
    if ens_id is None:
        return None
    fpath = os.path.join(TEMPS_DIR, f"Data {ens_id}.xlsx")
    if not os.path.isfile(fpath):
        return None
    try:
        df = pd.read_excel(fpath, sheet_name=DENS_SHEET)
    except Exception:
        return None
    if DENS_DATE_COL not in df.columns:
        return None
    dens_col = _find_density_col(df)
    if dens_col is None:
        return None
    ts = pd.to_datetime(df[DENS_DATE_COL], errors="coerce", utc=True)
    dens = pd.to_numeric(df[dens_col], errors="coerce")
    ok = ts.notna() & dens.notna()
    out = pd.DataFrame({"ts": ts[ok], "density": dens[ok]}).dropna().sort_values("ts")
    return out if not out.empty else None

def _align_density_to_sugar(
        sugar_df: pd.DataFrame,
        dens_df: pd.DataFrame) -> pd.Series:
    """
    Interpola densidad a timestamps de azúcar (sugar_df['timestamp']).
    Si no hay timestamp o dens_df vacío → devuelve NaN.
    """
    if sugar_df is None or dens_df is None:
        return pd.Series([np.nan]*len(sugar_df), index=sugar_df.index)
    if "timestamp" not in sugar_df.columns or sugar_df["timestamp"].notna().sum() == 0:
        return pd.Series([np.nan]*len(sugar_df), index=sugar_df.index)

    ts_sugar = pd.to_datetime(sugar_df["timestamp"], errors="coerce", utc=True)
    ts_sugar = ts_sugar.where(ts_sugar.notna())
    dens_df = dens_df.sort_values("ts")
    # Interpolación en eje de horas relativas
    t0 = min(ts_sugar.min(), dens_df["ts"].min())
    sug_h = (ts_sugar - t0).dt.total_seconds()/3600.0
    dens_h = (dens_df["ts"] - t0).dt.total_seconds()/3600.0
    # Evitar duplicados
    dens_h_vals, idx_unique = np.unique(dens_h.values, return_index=True)
    dens_vals = dens_df["density"].iloc[idx_unique].values
    interp = np.interp(
        np.clip(sug_h.values, dens_h_vals.min(), dens_h_vals.max()),
        dens_h_vals,
        dens_vals
    )
    return pd.Series(interp, index=sugar_df.index)

def _build_dataset(results_2025: Dict[str, pd.DataFrame],
                   sb2id: Dict[str, int]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for assay, df in results_2025.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if not {"Glucose","Fructose"}.issubset(df.columns):
            continue
        glu = pd.to_numeric(df["Glucose"], errors="coerce")
        fru = pd.to_numeric(df["Fructose"], errors="coerce")
        sugar = glu + fru
        # Requiere timestamp para alinear densidad
        if "timestamp" not in df.columns:
            continue
        dens_df = _load_density_table(assay, sb2id)
        dens_interp = _align_density_to_sugar(df, dens_df) if dens_df is not None else pd.Series([np.nan]*len(df))
        data_local = pd.DataFrame({
            "assay": assay,
            "time_h": df.get("time_hours", df.get("time_days", pd.Series([np.nan]*len(df)))*24.0),
            "timestamp": pd.to_datetime(df["timestamp"], errors="coerce", utc=True),
            "density": dens_interp,
            "glucose": glu,
            "fructose": fru,
            "total_sugar": sugar
        })
        data_local = data_local.dropna(subset=["density","total_sugar"])
        if data_local.shape[0] >= MIN_POINTS_PER_ASSAY:
            rows.append(data_local)
    if not rows:
        return pd.DataFrame(columns=["assay","time_h","density","total_sugar","glucose","fructose","timestamp"])
    return pd.concat(rows, ignore_index=True)

def _safe_rmse(y_true, y_pred):
    """
    Calcula RMSE compatible con versiones antiguas de sklearn
    que no aceptan el argumento 'squared'.
    """
    try:
        # versiones modernas
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # fallback manual
        return np.sqrt(mean_squared_error(y_true, y_pred))

def fit_sugar_density_model(df: pd.DataFrame,
                            degree: int = POLY_DEGREE):
    X = df[["density"]].values
    y = df["total_sugar"].values
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linreg", LinearRegression())
    ])
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = _safe_rmse(y, y_pred)
    metrics = {
        "R2": r2_score(y, y_pred),
        "RMSE": rmse,
        "MAE": mean_absolute_error(y, y_pred),
        "sklearn_version": sklearn.__version__
    }
    return model, y_pred, metrics

def plot_results(df: pd.DataFrame,
                 y_pred: np.ndarray,
                 model,
                 degree: int):
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    ax1, ax2, ax3 = axes

    # Scatter density vs total_sugar con curva
    ax1.scatter(df["density"], df["total_sugar"], c="tab:blue", s=18, alpha=0.6, label="Datos")
    # Curva suave
    dens_grid = np.linspace(df["density"].min(), df["density"].max(), 300).reshape(-1,1)
    sugar_grid = model.predict(dens_grid)
    ax1.plot(dens_grid, sugar_grid, "r", lw=2, label=f"Modelo poly deg={degree}")
    ax1.set_xlabel("Densidad")
    ax1.set_ylabel("Azúcar total (Glucosa+Fructosa)")
    ax1.set_title("Curva ajuste azúcar vs densidad")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Predicho vs real
    ax2.scatter(df["total_sugar"], y_pred, c="tab:green", s=18, alpha=0.6)
    lims = [min(df["total_sugar"].min(), y_pred.min()),
            max(df["total_sugar"].max(), y_pred.max())]
    ax2.plot(lims, lims, "k--", lw=1)
    ax2.set_xlabel("Azúcar total real")
    ax2.set_ylabel("Azúcar total predicho")
    ax2.set_title("Predicho vs Real")
    ax2.grid(alpha=0.3)

    # Residuales
    residuals = df["total_sugar"].values - y_pred
    ax3.scatter(df["density"], residuals, c="tab:orange", s=18, alpha=0.6)
    ax3.axhline(0, color="k", lw=1)
    ax3.set_xlabel("Densidad")
    ax3.set_ylabel("Residual (real - pred)")
    ax3.set_title("Residuales vs Densidad")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def per_assay_metrics(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    out_rows = []
    arr_pred = y_pred
    for assay, sub in df.groupby("assay"):
        idx = sub.index
        y_true = sub["total_sugar"].values
        y_p = arr_pred[idx]
        out_rows.append({
            "assay": assay,
            "n": len(sub),
            "R2": r2_score(y_true, y_p) if len(sub) > 1 else np.nan,
            "RMSE": _safe_rmse(y_true, y_p),  # <-- corregido (antes mean_squared_error(..., squared=False))
            "MAE": mean_absolute_error(y_true, y_p)
        })
    return pd.DataFrame(out_rows).sort_values("assay")

def main(args):
    # Cargar resultados 2025
    try:
        import Calibration_data_preprocess as cdp
    except ImportError as e:
        raise SystemExit(f"No se puede importar Calibration_data_preprocess: {e}")

    results_2025, _ = cdp.process_all()
    # Obtener mapping SB2ID
    from Calibration_data_preprocess import SB2ID as MAP_SB2ID

    dataset = _build_dataset(results_2025, MAP_SB2ID)
    if dataset.empty:
        print("[SUGAR-DENS] Dataset vacío (verifique densidades y timestamps).")
        return

    print(f"[SUGAR-DENS] Filas totales dataset: {len(dataset)} | Ensayos: {dataset['assay'].nunique()}")
    print(dataset.groupby("assay").size().rename("rows"))

    model, y_pred, metrics = fit_sugar_density_model(dataset, degree=args.degree)
    print("\n[MÉTRICAS GLOBALES]")
    for k,v in metrics.items():
        if k == "sklearn_version":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")

    pm = per_assay_metrics(dataset, y_pred)
    print("\n[MÉTRICAS POR ENSAYO]")
    print(pm.to_string(index=False, justify="center", float_format=lambda x: f"{x:.4f}"))

    # Guardar modelo simple (coeficientes) y dataset
    coef_path = os.path.join(args.outdir, "sugar_density_model_coeffs.txt")
    os.makedirs(args.outdir, exist_ok=True)
    with open(coef_path, "w", encoding="utf-8") as f:
        pipe = model.named_steps
        scaler: StandardScaler = pipe["scaler"]
        poly: PolynomialFeatures = pipe["poly"]
        lin: LinearRegression = pipe["linreg"]
        f.write("Polynomial regression (density -> total_sugar)\n")
        f.write(f"Degree: {args.degree}\n")
        f.write(f"Feature names: {poly.get_feature_names_out(['density']).tolist()}\n")
        f.write(f"Scaler mean: {scaler.mean_.tolist()} var: {scaler.var_.tolist()}\n")
        f.write(f"Coefficients: {lin.coef_.tolist()}\n")
        f.write(f"Intercept: {lin.intercept_}\n")
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.number)):
                f.write(f"{k}: {float(v):.6f}\n")
            else:
                f.write(f"{k}: {v}\n")
    dataset.to_csv(os.path.join(args.outdir, "sugar_density_dataset.csv"), index=False)
    pm.to_csv(os.path.join(args.outdir, "sugar_density_per_assay_metrics.csv"), index=False)
    print(f"\n[SUGAR-DENS] Guardado modelo (coef) en: {coef_path}")

    plot_results(dataset, y_pred, model, args.degree)

def _parse_args():
    ap = argparse.ArgumentParser(description="Calibración modelo azúcar total vs densidad (ensayos 2025).")
    ap.add_argument("--degree", type=int, default=POLY_DEGREE, help="Grado polinomio.")
    ap.add_argument("--outdir", default="sugar_density_out", help="Carpeta salida.")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    main(args)
