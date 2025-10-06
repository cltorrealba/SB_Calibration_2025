import argparse
import os
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np

# ---------------- Core builders ----------------
def load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"assay","year","T_bin","tnut","condition_id","batch_id","replicate_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    return df

def build_group_table(df: pd.DataFrame) -> pd.DataFrame:
    def _tnut_med(s: pd.Series):
        vals = pd.to_numeric(s, errors="coerce").dropna()
        return float(vals.median()) if len(vals) else np.nan
    g = (df.groupby(["year","batch_id","condition_id"], dropna=False)
           .agg(
               T_bin=("T_bin", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
               tnut_med=("tnut", _tnut_med),
               n_rep=("replicate_id","nunique"),
               assays=("assay", lambda s: tuple(sorted(set(s))))
           )
           .reset_index())
    # Distancias para diagnóstico (llenadas por año luego)
    return g

# ---------------- Selection helpers (reemplazados) ----------------
def _pick_one_per_bin(groups: pd.DataFrame, already: set, year_med: float) -> List[int]:
    picks = []
    # Ordenar bins por escasez (menos grupos primero)
    bin_order = (groups.groupby("T_bin").size()
                          .sort_values(ascending=True).index.tolist())
    for b in bin_order:
        cand = groups[(groups["T_bin"] == b) & (~groups.index.isin(already))].copy()
        if cand.empty:
            continue
        cand["dist"] = (pd.to_numeric(cand["tnut_med"], errors="coerce") - year_med).abs()
        cand = cand.sort_values(
            by=["dist","batch_id","condition_id"],
            ascending=[False, True, True],
            na_position="last"
        )
        picks.append(cand.index[0])
        already.add(cand.index[0])
    return picks

def _ensure_early_late(groups: pd.DataFrame, already: set,
                       p25: float, p75: float, target_extra: int) -> List[int]:
    adds = []
    tnut = pd.to_numeric(groups["tnut_med"], errors="coerce")
    def _pick_one(mask):
        cand = groups[mask & (~groups.index.isin(already))].copy()
        if cand.empty:
            return None
        cand["dist_edge"] = (tnut.loc[cand.index] - np.nanmedian(tnut)).abs()
        cand = cand.sort_values(
            by=["dist_edge","batch_id","condition_id"],
            ascending=[False, True, True],
            na_position="last"
        )
        return cand.index[0]
    early_mask = tnut <= p25
    late_mask  = tnut >= p75
    have_early = any(i in already for i in groups.index[early_mask])
    have_late  = any(i in already for i in groups.index[late_mask])
    if not have_early:
        pick = _pick_one(early_mask)
        if pick is not None:
            adds.append(pick); already.add(pick)
    if not have_late and len(adds) < target_extra:
        pick = _pick_one(late_mask)
        if pick is not None:
            adds.append(pick); already.add(pick)
    return adds

def _diversity_completion(groups: pd.DataFrame, already: set, n_target: int) -> List[int]:
    remaining = []
    if len(already) >= n_target:
        return remaining
    rest = groups.loc[~groups.index.isin(already)].copy()
    if rest.empty:
        return remaining
    med = pd.to_numeric(groups["tnut_med"], errors="coerce").median()
    picks_df = groups.loc[list(already)] if already else pd.DataFrame(columns=groups.columns)
    bin_counts_picked = picks_df["T_bin"].value_counts() if not picks_df.empty else pd.Series(dtype=int)
    rest["div_score"] = (pd.to_numeric(rest["tnut_med"], errors="coerce") - med).abs()
    rest["bin_pen"] = rest["T_bin"].map(bin_counts_picked).fillna(0)
    rest = rest.sort_values(
        by=["bin_pen","div_score","batch_id","condition_id"],
        ascending=[True, False, True, True],
        na_position="last"
    )
    for idx in rest.index:
        if len(already) >= n_target:
            break
        already.add(idx)
        remaining.append(idx)
    return remaining

def select_validation_groups(groups: pd.DataFrame,
                             year: int,
                             n_valid_target: int,
                             require_all_bins: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    g_y = groups[groups["year"] == year].copy()
    if g_y.empty:
        return pd.DataFrame(), pd.DataFrame()
    if n_valid_target >= len(g_y):
        # Edge case: everything becomes validation
        return g_y, pd.DataFrame()
    # Estadísticos tnut
    tnut_vals = pd.to_numeric(g_y["tnut_med"], errors="coerce").dropna()
    year_med = float(tnut_vals.median()) if len(tnut_vals) else np.nan
    p25 = float(tnut_vals.quantile(0.25)) if len(tnut_vals) else np.nan
    p75 = float(tnut_vals.quantile(0.75)) if len(tnut_vals) else np.nan

    selected: set = set()
    # 1. Uno por bin (si se exige)
    if require_all_bins:
        _pick_one_per_bin(g_y, selected, year_med)

    # 2. Asegurar early / late si hay datos suficientes
    if len(selected) < n_valid_target and np.isfinite(p25) and np.isfinite(p75):
        _ensure_early_late(g_y, selected, p25, p75, n_valid_target - len(selected))

    # 3. Completar hasta target
    if len(selected) < n_valid_target:
        _diversity_completion(g_y, selected, n_valid_target)

    # 4. Ajuste final (por si se pasó)
    if len(selected) > n_valid_target:
        # Mantener primero los seleccionados por orden reproducible
        selected = set(list(selected)[:n_valid_target])

    valid_groups = g_y.loc[list(selected)].copy()
    train_groups = g_y.loc[~g_y.index.isin(selected)].copy()
    valid_groups["split_reason"] = "validation"
    train_groups["split_reason"] = "train"
    return valid_groups, train_groups

# ---------------- Orchestrator ----------------
def partition_metadata(meta_path: str,
                       out_dir: str,
                       n_valid_2025: int = 3,
                       n_valid_2024: int = 5,
                       strict_bins: bool = True,
                       seed: int = 42):
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    meta = load_metadata(meta_path)
    groups = build_group_table(meta)

    valid25, train25 = select_validation_groups(groups, 2025, n_valid_2025, require_all_bins=strict_bins)
    valid24, train24 = select_validation_groups(groups, 2024, n_valid_2024, require_all_bins=strict_bins)

    groups_split = pd.concat([valid25, train25, valid24, train24], ignore_index=True)
    groups_split["split"] = groups_split["split_reason"]
    groups_split = groups_split.drop(columns=["split_reason"])

    # Expandir a nivel assay
    def _expand(group_rows: pd.DataFrame, label: str) -> List[str]:
        assays = []
        for _, r in group_rows.iterrows():
            assays.extend(list(r["assays"]))
        return sorted(set(assays))

    valid_assays = _expand(valid25, "valid") + _expand(valid24, "valid")
    train_assays = _expand(train25, "train") + _expand(train24, "train")

    # Guardar
    pd.Series(sorted(train_assays)).to_csv(os.path.join(out_dir, "train_ids.csv"), index=False, header=False)
    pd.Series(sorted(valid_assays)).to_csv(os.path.join(out_dir, "valid_ids.csv"), index=False, header=False)
    groups_split.to_csv(os.path.join(out_dir, "groups_split.csv"), index=False)

    # ---------- Nuevas salidas para etapa de calibración ----------
    # Mapa assay -> split con metadata original
    split_map = []
    split_lookup = {a: "valid" for a in valid_assays}
    split_lookup.update({a: "train" for a in train_assays})
    cols_keep = ["assay","year","T_bin","tnut","condition_id","batch_id","replicate_id"]
    meta_subset = meta[cols_keep].copy()
    meta_subset["split"] = meta_subset["assay"].map(split_lookup)
    assay_split_df = meta_subset.sort_values(["year","split","assay"]).reset_index(drop=True)
    # Guardar
    assay_split_path = os.path.join(out_dir, "assay_split.csv")
    assay_split_df.to_csv(assay_split_path, index=False)

    # Metadata separada por split
    train_meta_path = os.path.join(out_dir, "train_metadata.csv")
    valid_meta_path = os.path.join(out_dir, "valid_metadata.csv")
    assay_split_df[assay_split_df.split=="train"].to_csv(train_meta_path, index=False)
    assay_split_df[assay_split_df.split=="valid"].to_csv(valid_meta_path, index=False)

    print("\n[PARTITION] Archivos generados:")
    print(f"  - train_ids.csv / valid_ids.csv (listas de ensayos)")
    print(f"  - groups_split.csv (grupos (year,batch,condition) con atributos)")
    print(f"  - assay_split.csv (ensayo + split + metadata clave)")
    print(f"  - train_metadata.csv / valid_metadata.csv (metadata filtrada por split)")

    # Reporte
    print(f"[PARTITION] metadata={meta_path}")
    print(f"[PARTITION] Grupos totales: {len(groups)} | 2025={len(groups[groups.year==2025])} | 2024={len(groups[groups.year==2024])}")
    print(f"[PARTITION] Valid 2025 grupos={len(valid25)} target={n_valid_2025}  -> assays={len(valid_assays)} (parcial)")
    print(f"[PARTITION] Valid 2024 grupos={len(valid24)} target={n_valid_2024}")
    print(f"[PARTITION] Train assays={len(train_assays)} | Valid assays={len(valid_assays)}")
    # Cobertura T_bin por año en validación
    val_groups = pd.concat([valid25, valid24])
    print("\n[PARTITION] Cobertura validación (year,T_bin):")
    if not val_groups.empty:
        print(val_groups.groupby(["year","T_bin"]).size().reset_index(name="n").to_string(index=False))
    else:
        print("Sin grupos de validación")

    # Early / late check
    for yr in [2025, 2024]:
        vg = val_groups[val_groups.year == yr]
        if vg.empty:
            continue
        tnut_valid = pd.to_numeric(groups[groups.year == yr].tnut_med, errors="coerce").dropna()
        if len(tnut_valid):
            p25, p75 = tnut_valid.quantile(0.25), tnut_valid.quantile(0.75)
            early_present = (pd.to_numeric(vg.tnut_med, errors="coerce") <= p25).any()
            late_present  = (pd.to_numeric(vg.tnut_med, errors="coerce") >= p75).any()
            print(f"[PARTITION] Año {yr}: early_ok={early_present} late_ok={late_present} (p25={p25:.2f}, p75={p75:.2f})")

    # Chequeo disjoint
    overlap = set(train_assays).intersection(valid_assays)
    if overlap:
        print(f"[PARTITION][WARN] {len(overlap)} assays en ambos splits: {sorted(list(overlap))[:10]}...")

    # Aviso bins faltantes si strict_bins
    if strict_bins:
        for yr in [2024, 2025]:
            bins_year = set(groups[groups.year==yr]["T_bin"].dropna().unique().tolist())
            if bins_year:
                required = {"low","med","high"}
                missing = required - bins_year
                if missing:
                    print(f"[PARTITION][INFO] Año {yr}: faltan bins {sorted(missing)}; cobertura completa imposible.")

    # Resumen por año (valid)
    print("\n[PARTITION] Resumen por año (validación):")
    for yr, vg in [(2025, valid25), (2024, valid24)]:
        n_assays_valid = sum(len(a) for a in vg["assays"]) if not vg.empty else 0
        print(f"  {yr}: grupos_valid={len(vg)} assays_valid={n_assays_valid}")

    return {
        "metadata": meta,
        "groups": groups,
        "groups_split": groups_split,
        "train_assays": train_assays,
        "valid_assays": valid_assays,
        "assay_split": assay_split_df
    }

# ---------------- CLI ----------------
def _parse_args():
    ap = argparse.ArgumentParser(description="Partición inteligente train/valid basada en metadata.")
    ap.add_argument("--meta", default="metadata_2024_2025.csv", help="Ruta metadata CSV.")
    ap.add_argument("--outdir", default="splits", help="Carpeta salida.")
    ap.add_argument("--valid2025", type=int, default=3, help="Grupos validación 2025.")
    ap.add_argument("--valid2024", type=int, default=5, help="Grupos validación 2024.")
    ap.add_argument("--no-strict-bins", action="store_true", help="No exigir 1 grupo por T_bin.")
    ap.add_argument("--seed", type=int, default=42, help="Semilla.")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    partition_metadata(
        meta_path=args.meta,
        out_dir=args.outdir,
        n_valid_2025=args.valid2025,
        n_valid_2024=args.valid2024,
        strict_bins=not args.no_strict_bins,
        seed=args.seed
    )
