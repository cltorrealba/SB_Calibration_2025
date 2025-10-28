from __future__ import annotations
import os
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sb_calibration.preprocess.calibration_preprocess import ETHANOL_DENSITY_G_ML, SLOPE, INTERCEPT


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
        # Optional for initial biomass extraction (legacy-like)
        'Concentration': ('concentration', 'conc'),
        'Viability': ('viability', 'viable'),
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
            # Convert Alcohol/Ethanol % v/v to Ethanol g/L if needed
            if key == 'Ethanol':
                col_l = str(col).strip().lower()
                vm = val[np.isfinite(val)]
                # Heuristics:
                # 1) If column name suggests percent (alcohol, v/v, percent), convert
                name_suggests_pct = (('alcohol' in col_l) and ('ethanol' not in col_l)) or any(tok in col_l for tok in ('v/v','vv','percent','porcentaje','%'))
                # 2) If values look like fractions (<=1.1) -> fraction to percent
                looks_fraction = (vm.size and np.nanmedian(vm) <= 1.1)
                # 3) If values look like percent range (<=40) -> treat as percent
                looks_percent = (vm.size and not looks_fraction and np.nanmedian(vm) <= 40.0)
                if name_suggests_pct or looks_fraction or looks_percent:
                    if looks_fraction:
                        val = val * 100.0
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

    # If there are lab measurements extending beyond the operational grid, extend the grid
    lab_keys = ['YAN', 'Glucose', 'Fructose', 'Ethanol']
    lab_max_t = None
    for k in lab_keys:
        if k in sigs and not sigs[k].empty:
            tt = pd.to_numeric(sigs[k]['time_h'], errors='coerce').to_numpy(dtype=float)
            vv = pd.to_numeric(sigs[k][k], errors='coerce').to_numpy(dtype=float)
            m = ~(np.isnan(tt) | np.isnan(vv))
            if m.any():
                t_last = float(np.nanmax(tt[m]))
                lab_max_t = t_last if lab_max_t is None else max(lab_max_t, t_last)
    if lab_max_t is not None and (lab_max_t > (base_t.max() if base_t.size else -np.inf) + 1e-9):
        base_t = np.unique(np.append(base_t, lab_max_t))
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

    # Derive SugarTotal_exp (S from density) to be used in objective
    try:
        import pandas as _pd
        ds_path = os.path.join('sugar_density_out', 'sugar_density_dataset.csv')
        mat['SugarTotal_exp'] = np.nan
        if os.path.exists(ds_path):
            dsd = _pd.read_csv(ds_path)
            # Preferred: use per-assay records if present
            sub = None
            if 'assay' in dsd.columns:
                sub = dsd[dsd['assay'].astype(str) == str(assay_id)]
            if sub is not None and not sub.empty and ('time_h' in sub.columns) and ('total_sugar' in sub.columns):
                tds = _pd.to_numeric(sub['time_h'], errors='coerce').to_numpy(dtype=float)
                sds = _pd.to_numeric(sub['total_sugar'], errors='coerce').to_numpy(dtype=float)
                m = ~(np.isnan(tds) | np.isnan(sds))
                if m.any():
                    s_on_grid = np.interp(base_t, tds[m], sds[m])
                    mat['SugarTotal_exp'] = s_on_grid
            else:
                # Fallback: fit global polynomial density->sugar and apply to Densidad column
                if ('density' in dsd.columns) and ('total_sugar' in dsd.columns):
                    den_global = _pd.to_numeric(dsd['density'], errors='coerce').to_numpy(dtype=float)
                    sug_global = _pd.to_numeric(dsd['total_sugar'], errors='coerce').to_numpy(dtype=float)
                    mg = ~(np.isnan(den_global) | np.isnan(sug_global))
                    if mg.any():
                        coef = np.polyfit(den_global[mg], sug_global[mg], deg=3)
                        den_local = mat['Densidad'].to_numpy(dtype=float)
                        sl = np.polyval(coef, den_local)
                        mat['SugarTotal_exp'] = sl
        # ensure non-negative
        if 'SugarTotal_exp' in mat.columns:
            mat['SugarTotal_exp'] = mat['SugarTotal_exp'].astype(float)
            mat.loc[~np.isfinite(mat['SugarTotal_exp']), 'SugarTotal_exp'] = np.nan
            mat['SugarTotal_exp'] = np.clip(mat['SugarTotal_exp'], a_min=0.0, a_max=None)
    except Exception:
        # leave SugarTotal_exp as NaN if anything fails
        pass

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
        # choose initial >=0 and final = latest available (do NOT restrict by t_end)
        t_init = None; v_init = None
        pos = tt[tt >= 0.0]
        if pos.size:
            idx = int(np.argmin(pos))
            t_init = float(pos[idx])
            # value at that time
            v_init = float(vv[tt >= 0.0][idx])
        t_final = None; v_final = None
        # final = last valid measurement time/value
        idx2 = int(np.argmax(tt))
        t_final = float(tt[idx2])
        v_final = float(vv[idx2])
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

    # Derive and inject initial biomass viable using Concentration (+ Viability if available), legacy-like
    try:
        if 'Concentration' in sigs and not sigs['Concentration'].empty:
            dC = sigs['Concentration']
            tt = dC['time_h'].to_numpy(dtype=float)
            vv = pd.to_numeric(dC['Concentration'], errors='coerce').to_numpy(dtype=float)
            m = ~(np.isnan(tt) | np.isnan(vv))
            if m.any():
                tt = tt[m]; vv = vv[m]
                # initial at earliest non-negative time
                pos = tt[tt >= 0.0]
                if pos.size:
                    idx = int(np.argmin(pos))
                    t_init = float(pos[idx])
                    C0 = float(vv[tt >= 0.0][idx])
                    # total biomass from calibration
                    total_gL = max(0.0, SLOPE * C0 + INTERCEPT)
                    # viability fraction if available
                    f_viab = 0.5
                    if 'Viability' in sigs and not sigs['Viability'].empty:
                        dV = sigs['Viability']
                        tv = dV['time_h'].to_numpy(dtype=float)
                        vv2 = pd.to_numeric(dV['Viability'], errors='coerce').to_numpy(dtype=float)
                        mv = ~(np.isnan(tv) | np.isnan(vv2))
                        if mv.any():
                            tv = tv[mv]; vv2 = vv2[mv]
                            # nearest viability value to t_init
                            j = int(np.argmin(np.abs(tv - t_init)))
                            vraw = float(vv2[j])
                            # interpret as fraction if <=1, else percent [0,100]
                            f_viab = vraw if (0.0 <= vraw <= 1.0) else (max(0.0, min(100.0, vraw)) / 100.0)
                    X0_viab = total_gL * f_viab
                    # set at nearest grid point
                    i0 = int(np.argmin(np.abs(base_t - t_init)))
                    if np.isnan(mat.loc[i0, 'biomass_viable_gL']) or mat.loc[i0, 'biomass_viable_gL'] <= 0:
                        mat.loc[i0, 'biomass_viable_gL'] = X0_viab
    except Exception:
        pass

    # Try to read explicit 'descube' endpoints from Laboratorio (legacy behavior)
    try:
        lab = pd.read_excel(xlsx_path, sheet_name='Laboratorio')
        # Normalize column names (lowercase, remove accents, replace non-alnum by underscore)
        def _norm_name(s: str) -> str:
            s = (s or '').strip().lower()
            try:
                s = pd.Series([s]).str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('ascii').iloc[0]
            except Exception:
                pass
            import re
            s = re.sub(r"[^a-z0-9]+", "_", s)
            s = s.strip('_')
            return s
        name_map = {_norm_name(c): c for c in lab.columns}
        def _find_col(*cands):
            for k in name_map.keys():
                for c in cands:
                    if c == k:
                        return name_map[k]
            # partial match
            for k in name_map.keys():
                for c in cands:
                    if c in k:
                        return name_map[k]
            return None
        mcol = _find_col('muestreo_text', 'muestreo', 'etapa')
        vname = _find_col('variable_text', 'template_variable_text', 'variable')
        vnum = _find_col('valor_numeric', 'valor_num', 'valor_numerico')
        vtxt = _find_col('valor')
        dcol = _find_col('create_fecha', 'fecha', 'fecha_muestra', 'datetime')
        if mcol and vname and (vnum or vtxt):
            m = lab[mcol].astype(str).str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('ascii')
            desc_all = lab.loc[m.str.contains('descube', na=False)].copy()
            if not desc_all.empty:
                # If there's a datetime column, keep only the latest 'descube' batch
                if dcol:
                    desc_all['_dt_'] = pd.to_datetime(desc_all[dcol], errors='coerce')
                    max_dt = desc_all['_dt_'].max()
                    if pd.notna(max_dt):
                        desc = desc_all.loc[desc_all['_dt_'] == max_dt]
                    else:
                        desc = desc_all
                else:
                    desc = desc_all
                # Map spanish names to canonical
                def _norm(s: str) -> str:
                    s = (s or '').strip().lower()
                    repl = (('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ñ','n'))
                    for a,b in repl:
                        s = s.replace(a,b)
                    s = s.replace('-', ' ')
                    return ' '.join(s.split())
                vals: Dict[str, float] = {}
                # Track if Ethanol came as 'alcohol' (% v/v) to force conversion
                ethanol_is_alcohol_flag = False
                for _, r in desc.iterrows():
                    vlabel_norm = _norm(str(r[vname]))
                    nm = None
                    if any(tok in vlabel_norm for tok in ("alcohol","ethanol")):
                        nm = 'Ethanol'
                    elif any(tok in vlabel_norm for tok in ("glucosa","glucose")):
                        nm = 'Glucose'
                    elif any(tok in vlabel_norm for tok in ("fructosa","fructose")):
                        nm = 'Fructose'
                    elif 'yan' in vlabel_norm:
                        nm = 'YAN'
                    if not nm:
                        continue
                    val = None
                    if vnum and (vnum in r) and pd.notna(r[vnum]):
                        try:
                            val = float(r[vnum])
                        except Exception:
                            val = None
                    if val is None and vtxt and (vtxt in r) and pd.notna(r[vtxt]):
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
                    if nm == 'Ethanol' and ('alcohol' in vlabel_norm):
                        ethanol_is_alcohol_flag = True
                if vals:
                    # Write to last grid row (descube cierre); overwrite to ensure correct units
                    i_last = len(mat) - 1
                    # Ethanol: if came as 'alcohol' treat as % v/v; handle fraction case
                    if 'Ethanol' in vals:
                        e_val = float(vals['Ethanol'])
                        # Heuristics: convert to g/L if value looks like fraction (<=1.1) or percent (<=40),
                        # regardless of label; if explicitly from 'alcohol', definitely convert.
                        if ethanol_is_alcohol_flag or (e_val <= 1.1) or (e_val <= 40.0):
                            if e_val <= 1.1:
                                e_val *= 100.0  # fraction -> percent
                            e_gl = max(0.0, e_val) * ETHANOL_DENSITY_G_ML * 10.0
                        else:
                            # assume already in g/L
                            e_gl = max(0.0, e_val)
                        mat.loc[i_last, 'Ethanol'] = e_gl
                    for k in ('YAN','Glucose','Fructose'):
                        if k in vals:
                            mat.loc[i_last, k] = float(vals[k])
    except Exception:
        pass

    # Canonical column order (includes Densidad as in legacy examples)
    cols = ['time_h', 'biomass_viable_gL', 'YAN', 'Glucose', 'Fructose', 'Ethanol', 'SugarTotal_exp', 'Temperature_C', 'Densidad']
    # Keep any extra columns at the end
    front = [c for c in cols if c in mat.columns]
    rest = [c for c in mat.columns if c not in front]
    mat = mat[front + rest]
    return mat
