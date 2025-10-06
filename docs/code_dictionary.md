# Diccionario de códigos y mapeos — SB_Calibration_2025

Este documento centraliza los códigos relevantes usados en los pipelines y cómo
interpretarlos. Está pensado como referencia rápida para desarrolladores y
personas que integran planillas experimentales.

## Códigos de ensayo (SBxxx) y archivos asociados

- SB003 … SB012: identificadores de ensayos. En la carpeta `Datos Experimentales/`
  aparecen archivos tipo `Data <ID>.xlsx` donde `<ID>` corresponde a un número
  mapeado a cada `SBxxx`.

### Mapeo SB -> Data ID (2025)

El mapeo actual aparece en `Calibration_data_preprocess.py` (constante `SB2ID`).
Resumen:

- SB003 -> 25026
- SB004 -> 25027
- SB005 -> 25028
- SB006 -> 25029
- SB007 -> 25085
- SB008 -> 25086
- SB009 -> 25150
- SB010 -> 25151
- SB011 -> 25170
- SB012 -> 25171

Si añades nuevos ensayos, actualiza `SB2ID` (o, preferible, el DataLoader deberá
permitir una fuente de mapeo externa y no hardcodear el diccionario).

## Nombres de columnas canónicas
En los pipelines se normalizan múltiples nombres a un set canónico. Utiliza estos
cuando crees planillas o transformaciones.

- time_h: tiempo en horas desde t0
- timestamp / Fecha y hora: timestamp original del muestreo
- biomass_viable_gL: biomasa viable (g/L)
- biomass_total_gL: biomasa total (g/L)
- YAN: nitrógeno asimilable (mg/L experimental)
- Glucose: glucosa (g/L)
- Fructose: fructosa (g/L)
- Ethanol / Alcohol: etanol (g/L) — ojo: en algunas hojas viene en % v/v
- Temperature_C: temperatura en °C (cuando existe)
- Densidad / density: densidad (unit): utilizada para estimar azúcar total
- SugarTotal_exp: azúcar total experimental (si está disponible)

## Módulo `sugar_density`

El módulo `src/sb_calibration/preprocess/sugar_density.py` ofrece utilidades para
ajustar una función densidad -> azúcar. La función `fit_density_model(df, x_col, y_col)`
espera columnas con el porcentaje de azúcar (`sugar_pct` o similar) y densidad (`density_g_mL`).
Usa un pipeline de `PolynomialFeatures` + `LinearRegression` y devuelve el pipeline
entrenado y los coeficientes.

## Columnas derivadas y convenciones internas

- Internamente el objetivo normaliza N a g/L (se aplica `N_SCALE = 1e-3` para convertir mg/L -> g/L).
- `DEFAULT_X0` = [X0_biomass, N0_gL, G0, F0, E0] (valores por defecto para simulación).
- SSE normalizada por std global por variable (X,N,G,F,E); cuando una variable no está disponible, no contribuye.

## Modos de Jacobiano en la integración

- analytic: usa `zenteno_jacobian` y sparsidad `J_SPARSE` (más rápido/estable).
- numeric: calcula derivadas con complex-step y fallback a forward-diff (útil para validar).
- none: integra sin jacobiano explícito (más robusto, algo más lento).

Control desde CLI: `--jacobian analytic|numeric|none` y método `--method Radau|BDF`.

## Códigos de insumos / adiciones químicas

Durante el preprocesado / extracción de `chem_df` se buscan insumos relevantes:

- FDA: adición inorgánica (ej. fertilizante nitrogenado)
- Vitaferm: adición orgánica

Los parsers intentan extraer `time_h` y `valor` (cantidad, convertida a mg o g/L
según volumen). Si la hoja no contiene `time_h`, se intenta inferir desde `timestamp`.

Builder de pulsos (nuevo):
- `calibration.pulses.build_pulses_from_chem(chem_df)` devuelve `{assay: [(t_h, dN_gL), ...]}`
  detectando columnas flexibles (código con `SBxxx`, `YAN`, `time_h` o `timestamp`).
  Calcula deltas por fila (no negativos) en mg/L y los convierte a g/L para las integraciones.

Pesos por variable (nuevo):
- La función objetivo acepta `weights={"X":wX,"N":wN,"G":wG,"F":wF,"E":wE}`.
- La CLI expone flags `--w-x --w-n --w-g --w-f --w-e` y los propaga al optimizador/objetivo.

## Notas sobre naming inconsistent y consejos

- Evita acentos en nombres de columnas nuevas (usar `Temperature_C`, `Glucose`)
- Cuando exportes planillas a usar por los pipelines, incluye `time_h` o `timestamp`.
- Para densidad, colocar siempre columna exacta `Densidad` o `density` facilita
  la integración con `sugar_density`.
