from __future__ import annotations
import os
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sb_calibration.preprocess.calibration_preprocess import ETHANOL_DENSITY_G_ML


def _col_like(df: pd.DataFrame, *cands: str) -> Optional[str]:
    for c in df.columns:
        l = str(c).strip().lower()
        for cand in cands:
            if l == cand.lower():
                return c
    for c in df.columns:
        l = str(c).strip().lower()
        for cand in cands:
            if cand.lower() in l:
                return c
    return None


def _build_time_h_from_any(df: pd.DataFrame) -> Optional[np.ndarray]:
    # Try 'time_h' directly
    c = _col_like(df, 'time_h', 'tiempo_h', 'horas', 't_h')
    if c is not None:
        return pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)
    # Try timestamp
    ts_c = _col_like(df, 'timestamp', 'fecha', 'datetime', 'fecha_muestra')
    if ts_c is not None:
        ts = pd.to_datetime(df[ts_c], errors='coerce')
        if ts.notna().any():
            t0 = ts.min()
            return ((ts - t0).dt.total_seconds() / 3600.0).to_numpy(dtype=float)
    return None


def _read_temperature_table(xlsx_path: str) -> Optional[pd.DataFrame]:
    try:
        # Try common sheet name used in legacy
        dfT = pd.read_excel(xlsx_path, sheet_name='Manual Temperaturas')
    except Exception:
        # Fallback: first sheet that contains a temp-like column
        try:
            xl = pd.ExcelFile(xlsx_path)
            for s in xl.sheet_names:
                df = xl.parse(s)
                if _col_like(df, 'temperatura', 'temperature_c', 'temp_c') is not None:
                    dfT = df
                    break
            else:
                return None
        except Exception:
            return None

    c_temp = _col_like(dfT, 'temperatura', 'temperature_c', 'temp_c')
    if c_temp is None:
        return None
    t = _build_time_h_from_any(dfT)
    if t is None:
        return None
    temp_c = pd.to_numeric(dfT[c_temp], errors='coerce').to_numpy(dtype=float)
    m = ~(np.isnan(t) | np.isnan(temp_c))
    if not m.any():
        return None
    df = pd.DataFrame({'time_h': t[m], 'Temperature_C': temp_c[m]})
    df = df.sort_values('time_h').drop_duplicates(subset=['time_h'])
    return df


def _read_density_table(xlsx_path: str) -> Optional[pd.DataFrame]:
    """Read manual densities sheet if present and return time_h and Densidad (kg/m3-style as numeric)."""
    try:
        dfD = pd.read_excel(xlsx_path, sheet_name='Manual Densidades')
    except Exception:
        try:
            xl = pd.ExcelFile(xlsx_path)
            for s in xl.sheet_names:
                df = xl.parse(s)
                if _col_like(df, 'densidad', 'density') is not None:
                    dfD = df
                    break
            else:
                return None
        except Exception:
            return None

    c_den = _col_like(dfD, 'densidad', 'density')
    if c_den is None:
        return None
    t = _build_time_h_from_any(dfD)
    if t is None:
        return None
    den = pd.to_numeric(dfD[c_den], errors='coerce').to_numpy(dtype=float)
    m = ~(np.isnan(t) | np.isnan(den))
    if not m.any():
        return None
    df = pd.DataFrame({'time_h': t[m], 'Densidad': den[m]})
    df = df.sort_values('time_h').drop_duplicates(subset=['time_h'])
    return df


def _read_signals_any(xlsx_path: str) -> Dict[str, pd.DataFrame]:
    """Scan sheets to find columns for YAN, Glucose, Fructose, Ethanol.

    Returns per-signal dataframes with time_h and value, but DOES NOT resample/interpolate.
    If a column that maps to 'Ethanol' is actually 'Alcohol' in % v/v, it is converted to g/L
    using ETHANOL_DENSITY_G_ML.
    """
    out: Dict[str, pd.DataFrame] = {}
    try:
        xl = pd.ExcelFile(xlsx_path)
    except Exception:
        return out
    targets = {
        'YAN': ('YAN',),
        'Glucose': ('glucose', 'glucosa'),
        'Fructose': ('fructose', 'fructosa'),
        'Ethanol': ('ethanol', 'alcohol'),
    }
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
        except Exception:
            continue
        t = _build_time_h_from_any(df)
        if t is None:
            continue
        for key, cands in targets.items():
            if key in out:
                continue
            col = _col_like(df, *cands)
            if col is None:
                continue
            val = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
            # Convert Alcohol % v/v to Ethanol g/L if needed
            if key == 'Ethanol' and str(col).strip().lower() in ('alcohol',):
                # % v/v to g/L: pct * density(g/mL) * 10
                val = np.clip(val, a_min=0.0, a_max=None) * ETHANOL_DENSITY_G_ML * 10.0
            m = ~(np.isnan(t) | np.isnan(val))
            if not m.any():
                continue
            dd = pd.DataFrame({'time_h': t[m], key: val[m]})
            dd = dd.sort_values('time_h').drop_duplicates(subset=['time_h'])
            out[key] = dd
        # Stop early if all found
        if set(out.keys()) == set(targets.keys()):
            break
    return out


def build_mats_for_assay_2024(assay_id: str, temps_dir: str = 'Datos Experimentales') -> Optional[pd.DataFrame]:
    """Build legacy-homologated mats for a 24xxx assay from its experimental Excel.

    Rules (matching legacy):
    - Time grid comes from the union of Temperature and Density measurement times.
    - Operational signals (Temperature_C, Densidad) are linearly interpolated onto the grid.
    - Lab signals (YAN [mg/L], Glucose, Fructose, Ethanol [g/L]) are NOT interpolated; only two points are injected:
        initial (first t≥0 present) and final (last t≤t_end present). All other grid rows remain NaN for lab vars.
    """
    xlsx_path = os.path.join(temps_dir, f"Data {assay_id}.xlsx")
    if not os.path.exists(xlsx_path):
        return None

    dfT = _read_temperature_table(xlsx_path)
    dfD = _read_density_table(xlsx_path)
    sigs = _read_signals_any(xlsx_path)

    # Build base time grid from operational signals
    t_parts = []
    if dfT is not None and not dfT.empty:
        t_parts.append(dfT['time_h'].to_numpy(dtype=float))
    if dfD is not None and not dfD.empty:
        t_parts.append(dfD['time_h'].to_numpy(dtype=float))
    if not t_parts:
        # fallback: use any signal time (prefers YAN) but clip to >=0
        for k in ('YAN', 'Glucose', 'Fructose', 'Ethanol'):
            if k in sigs:
                t_parts.append(sigs[k]['time_h'].to_numpy(dtype=float))
                break
        if not t_parts:
            return None
    base_t = np.unique(np.clip(np.concatenate(t_parts).astype(float), a_min=0.0, a_max=None))
    mat = pd.DataFrame({'time_h': base_t})

    # Operational interpolation
    if dfT is not None and not dfT.empty:
        mat['Temperature_C'] = np.interp(base_t, dfT['time_h'].to_numpy(dtype=float), dfT['Temperature_C'].to_numpy(dtype=float))
    else:
        mat['Temperature_C'] = np.nan
    if dfD is not None and not dfD.empty:
        mat['Densidad'] = np.interp(base_t, dfD['time_h'].to_numpy(dtype=float), dfD['Densidad'].to_numpy(dtype=float))
    else:
        mat['Densidad'] = np.nan

    # Prepare lab columns (NaN except initial/final injections)
    for c in ['biomass_viable_gL', 'biomass_dead_gL', 'YAN', 'AMMONIA', 'PAN', 'Fructose', 'Glucose', 'Glycerol', 'Ethanol']:
        if c not in mat.columns:
            mat[c] = np.nan

    # Inject only first and last observed values for lab signals
    t_end = float(base_t.max()) if base_t.size else np.nan
    def _inject_two_points(col: str, df_sig: pd.DataFrame):
        if df_sig is None or df_sig.empty:
            return
        tt = df_sig['time_h'].to_numpy(dtype=float)
        vv = df_sig[col].to_numpy(dtype=float)
        m = ~(np.isnan(tt) | np.isnan(vv))
        if not m.any():
            return
        tt = tt[m]; vv = vv[m]
        # choose initial >=0 and final <= t_end
        t_init = None; v_init = None
        pos = tt[tt >= 0.0]
        if pos.size:
            idx = int(np.argmin(pos))
            t_init = float(pos[idx])
            # value at that time
            v_init = float(vv[tt >= 0.0][idx])
        t_final = None; v_final = None
        if np.isfinite(t_end):
            le = tt[tt <= t_end]
            if le.size:
                idx2 = int(np.argmax(le))
                t_final = float(le[idx2])
                v_final = float(vv[tt <= t_end][idx2])
        # map to nearest grid indices and set values
        if t_init is not None:
            i0 = int(np.argmin(np.abs(base_t - t_init)))
            mat.loc[i0, col] = v_init
        if t_final is not None:
            i1 = int(np.argmin(np.abs(base_t - t_final)))
            mat.loc[i1, col] = v_final

    for key in ('YAN', 'Glucose', 'Fructose', 'Ethanol'):
        if key in sigs:
            _inject_two_points(key, sigs[key])

    # Try to read explicit 'descube' endpoints from Laboratorio (legacy behavior)
    try:
        lab = pd.read_excel(xlsx_path, sheet_name='Laboratorio')
        cols = {str(c).strip().lower(): c for c in lab.columns}
        mcol = cols.get('muestreo_text') or cols.get('muestreo') or cols.get('etapa')
        vname = cols.get('variable_text') or cols.get('template_variable_text') or cols.get('variable')
        vnum = cols.get('valor_numeric') or cols.get('valor_num')
        vtxt = cols.get('valor')
        if mcol and vname and (vnum or vtxt):
            m = lab[mcol].astype(str).str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('ascii')
            desc = lab.loc[m.str.contains('descube', na=False)].copy()
            if not desc.empty:
                # Take first occurrence like legacy
                if 'create_fecha' in lab.columns or 'fecha' in lab.columns:
                    # optional sort by time if present
                    dtc = 'create_fecha' if 'create_fecha' in lab.columns else 'fecha'
                    desc['_dt_'] = pd.to_datetime(desc[dtc], errors='coerce')
                    desc = desc.sort_values('_dt_')
                desc = desc.head(1)
                # Map spanish names to canonical
                def _norm(s: str) -> str:
                    s = (s or '').strip().lower()
                    repl = (('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ñ','n'))
                    for a,b in repl:
                        s = s.replace(a,b)
                    s = s.replace('-', ' ')
                    return ' '.join(s.split())
                MAP = {
                    'yan': 'YAN', 'alcohol': 'Ethanol', 'ethanol': 'Ethanol',
                    'glucosa': 'Glucose', 'glucose': 'Glucose', 'fructosa': 'Fructose', 'fructose': 'Fructose'
                }
                vals: Dict[str, float] = {}
                for _, r in desc.iterrows():
                    nm = MAP.get(_norm(str(r[vname])), None)
                    if not nm:
                        continue
                    val = None
                    if vnum and pd.notna(r.get(vnum)):
                        try:
                            val = float(r[vnum])
                        except Exception:
                            val = None
                    if val is None and vtxt and pd.notna(r.get(vtxt)):
                        import re
                        s = str(r[vtxt]).replace(',', '.')
                        m0 = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                        if m0:
                            try:
                                val = float(m0.group(0))
                            except Exception:
                                val = None
                    if val is None:
                        continue
                    vals[nm] = float(val)
                if vals:
                    # Write to last grid row; convert Alcohol% to Ethanol g/L if needed
                    i_last = len(mat) - 1
                    if 'Ethanol' in vals and mat.loc[i_last, 'Ethanol'] != mat.loc[i_last, 'Ethanol']:
                        # assume % v/v
                        mat.loc[i_last, 'Ethanol'] = max(0.0, float(vals['Ethanol'])) * ETHANOL_DENSITY_G_ML * 10.0
                    for k in ('YAN','Glucose','Fructose'):
                        if k in vals and mat.loc[i_last, k] != mat.loc[i_last, k]:
                            mat.loc[i_last, k] = float(vals[k])
    except Exception:
        pass

    # Canonical column order (includes Densidad as in legacy examples)
    cols = ['time_h', 'biomass_viable_gL', 'YAN', 'Glucose', 'Fructose', 'Ethanol', 'Temperature_C', 'Densidad']
    # Keep any extra columns at the end
    front = [c for c in cols if c in mat.columns]
    rest = [c for c in mat.columns if c not in front]
    mat = mat[front + rest]
    return mat
