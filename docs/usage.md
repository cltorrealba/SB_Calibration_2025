Migration note (2025-10-02)
---------------------------------
He movido los scripts monolíticos originales a `legacy/` para dejarlos como copia histórica y
# Guía de uso rápida — SB_Calibration_2025

Esta guía explica los comandos y flujos más comunes para preparar datos,
entrenar/calibrar el modelo y generar artefactos. Está orientada a desarrolladores
que usan el repositorio localmente.

Requisitos mínimos

- Entorno Python 3.11+ compatible con `requirements.txt`.
- Tener las planillas originales en:
  - `Procesos_I+D_2025_3.xlsx`
  - `Datos Experimentales/` (archivos `Data <ID>.xlsx`)

Instalación (recomendado: entorno virtual)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Flujos principales

1) Preprocesar datos (generar matrices por ensayo)

   - Script original (rápido):
     ```powershell
     python Calibration_data_preprocess.py
     python SW_Preprocess_data.py
     ```

   - Con la nueva fachada (cuando esté refactorizado):
     ```powershell
     python -m src.sb_calibration.cli.preprocess_cli --config configs/preprocess_dev.yaml
     ```

   Resultado: matrices guardadas en `mats/` (`assay=SBxxx.parquet`), `initial_conditions.csv` y `pulses_YAN.csv`.

2) Ajustar modelo densidad -> azúcar (solo 2025)

   ```powershell
   python sugar_density.py --outdir sugar_density_out
   ```

   Resultado: `sugar_density_model_coeffs.txt`, `sugar_density_dataset.csv`.

3) Calibrar el modelo (flujo principal)

   ```powershell
   python "Calibración global.py"
   ```

   (Próxima versión: `python -m src.sb_calibration.cli.calibrate_cli --config configs/calibrate_dev.yaml`)

Balance de la función objetivo (SSE)
------------------------------------

La SSE ahora soporta modos de balance para evitar que ensayos con más puntos dominen el ajuste.

- --sse-balance: per_assay (por defecto), per_point o none
   - per_assay: promedia la pérdida por variable dentro de cada ensayo; todos los ensayos pesan parecido.
   - per_point: suma las pérdidas por punto (histórico por defecto en legacy).
   - none: suma cruda de errores (no recomendado si hay distinta densidad temporal entre ensayos).

- --sse-resample-dt-h: submuestreo temporal opcional (en horas) antes de calcular la SSE para igualar densidades temporales; p.ej. 6.0.

Ejemplos (PowerShell):

```powershell
python -m src.sb_calibration.cli.calibrate_cli --file "Procesos_I+D_2025_3.xlsx" --sse-balance per_assay --sse-resample-dt-h 6.0 --plot --verbose
```

4) Validación

   - Tras calibrar, ejecutar validación opcional (se muestra desde el script principal).

Artefactos generados

- `mats/assay=SBxxx.parquet` o `.csv`
- `mats/pbest_checkpoint.npz` (checkpoint parámetros)
- `mats/initial_conditions.csv`, `mats/pulses_YAN.csv`, `mats/mats_index.csv`

Diagnóstico y debugging

- Usa la opción `verbose` en funciones clave (en la versión actual muchos prints ya existen).
- Para problemas de parsing de fechas: inspecciona la hoja `Manual Temperaturas` y columnas `medicion_fecha`.
- Si falla `solve_ivp` con stiff: revisar `ATOL/RTOL` y perfil de temperatura.

Contacto y próximos pasos

Migration note (2025-10-02)
---------------------------------
He movido los scripts monolíticos originales a `legacy/` para dejarlos como copia histórica y
limpiar la raiz del repositorio. Los archivos movidos son:

- `legacy/Calibración global.py`
- `legacy/Calibration_data_preprocess.py`
- `legacy/modelo_dinamico_sim.py`
- `legacy/SW_Preprocess_data.py`
- `legacy/sugar_density.py`
- `legacy/metadata.py`
- `legacy/data_partition.py`
- `legacy/Test_profiles.py`

Usa los nuevos módulos bajo `src/` (si están presentes) para desarrollo; los archivos en `legacy/`
no se ejecutan por defecto y están preservados como referencia.

Continuous Integration (CI) - qué es y por qué lo usamos
---------------------------------
CI significa "Continuous Integration" (Integración continua). En este repositorio hemos añadido
un flujo básico de CI que ejecuta pruebas automáticas cuando se hace push o se crea un pull request.
Beneficios principales:

- Verificar que los cambios no rompan la base de código (tests unitarios).
- Ejecutar linters y checks de formato de forma automatizada.
- Mantener checkpoints de build y validar compatibilidad de dependencias.

En `/.github/workflows/ci.yml` hay un workflow básico que instala dependencias y ejecuta `pytest`.
Podemos ampliar el workflow para incluir linting (`ruff`/`black`), comprobaciones de seguridad y
test de integración más avanzados.

Si quieres, puedo:

- Crear el skeleton `src/` y mover `modelo_dinamico_sim.py` como primer cambio (y añadir test de humo).
- Añadir un `pyproject.toml` con la información mínima para instalar en editable.

Ejecutar tests localmente
-------------------------

He añadido una configuración mínima de empaquetado (pyproject.toml + setup.cfg) y un `tests/conftest.py`
que añade el directorio del repositorio al `sys.path` durante la ejecución de pytest. Esto permite que
las pruebas importen el paquete usando la convención `src/...` sin necesidad de hacer `pip install -e .`.

Recomendación rápida (PowerShell):

- Abre tu entorno virtual y ejecuta:
   $env:PYTHONPATH = (Resolve-Path .).Path; pytest -q
- Alternativa: instala en editable mode desde la raíz del repo:
   pip install -e .

Estado actual del preprocesado (migración)
------------------------------------------

He migrado las partes iniciales del pipeline de preprocesado a `src/sb_calibration/preprocess/`.
Módulos actuales relevantes:

- `src/sb_calibration/preprocess/calibration_preprocess.py` — extracción por ensayo, correcciones
   isotónicas, construcción de matrices de calibración y funciones auxiliares.
- `src/sb_calibration/preprocess/sw_preprocess.py` — capa de compatibilidad con el script
   `SW_Preprocess_data.py` que delega a las funciones portadas.
- `src/sb_calibration/preprocess/sw_full.py` — utilidades SW específicas (p. ej. inferencia de inóculo)

Cómo usar las nuevas funciones (ejemplo rápido desde Python):

```python
from sb_calibration.preprocess import calibration_preprocess as cp
df_bdd = pd.read_excel('Procesos_I+D_2025_3.xlsx', sheet_name='BDD_Maestra')
results, combined = cp.process_all(file_path='Procesos_I+D_2025_3.xlsx')
results_with_T = cp.attach_temperature_to_results(results)
matrices = cp.build_calibration_matrices(results_with_T)
```

Próximo paso (en curso): incorporar la lectura completa de ficheros de temperaturas
(`Datos Experimentales/Data <ID>.xlsx`) en `sw_full.py` y añadir fixtures Excel para tests
de integración end-to-end.

Density -> Sugar (sugar_density)
--------------------------------

He añadido un módulo `src/sb_calibration/preprocess/sugar_density.py` con una API
simple (`fit_density_model`) que encapsula un pipeline PolynomialFeatures + LinearRegression
para ajustar densidad -> azúcar. Hay tests de humo en `tests/unit/test_sugar_density_smoke.py`.

Quick CSV usage (desde Python):

```python
from sb_calibration.preprocess import sugar_density as sd
pipeline, coefs = sd.run_from_csv('data/sugar_density_dataset.csv', 'sugar_density_model_coeffs.txt')
```

# Calibración dinámica MPCC (Julia) — Modo reducido, homotopía e inicialización

Esta sección documenta el flujo Julia (`MPCC_Zenteno.jl`) usado para una calibración relajada tipo MPCC con reducción estructural y homotopía en penalizaciones de complementariedad.

## Objetivos del pipeline

1. Disminuir tamaño (reacciones y filas de S) vía conjuntos pFBA/FVA (A, C, F) → modo reducido.
2. Sembrar valores iniciales consistentes (estados, flujos, multiplicadores, productos FO) para mejorar robustez.
3. Aplicar homotopía multi–etapa sobre (φ, w) para tensar la complementariedad gradualmente.
4. Registrar métricas basales y por etapa para comparar configuraciones y justificar parámetros.

## Archivos clave

| Archivo | Descripción |
|---------|-------------|
| `pfba_preprocess.jl` | Genera `results/reduced_sets.jld2` con A, C, F por FE. Debe ejecutarse antes si `REDUCED_MODE=1`. |
| `MPCC_Zenteno.jl` | Modelo JuMP: estados, flujos, penalizaciones, homotopía, escritura de métricas y plots. |
| `results/zenteno_metrics_baseline_*.txt` | Métricas antes de optimizar (tamaños, SSE0, PEN0, OBJ0, comp_max0, flags). |
| `results/zenteno_metrics_hom_sX_*.txt` | Métricas por etapa homotopía (phi, w, SSE, PEN, comp_max, stationarity_residual). |
| `results/zenteno_estimation_report_*.txt` | Resumen final (parámetros, FO_* stats, objetivo). |
| `zenteno_pre_ode_vs_data_*.png` / `zenteno_post_ode_vs_data_mpcc_*.png` | Comparación ODE inicial y final vs datos. |

## Variables de entorno (ENV)

| Var | Tipo | Default | Función |
|-----|------|---------|---------|
| `REDUCED_MODE` | {0,1} | 0 | Activa reconstrucción de un submodelo con sólo reacciones candidatas (A∪C) y filas activas de S. |
| `REDUCED_DISABLE_UPTAKE` | {0,1} | 0 | Elimina (si=1) desigualdades de uptake y sus FO asociados (para aislar comportamiento). |
| `PEN_REDUCED` | {0,1} | 1 | Si=1 suma penalizaciones sólo sobre candidatos C por FE; si=0 (legado) todas las reacciones. |
| `PEN_NONNEG` | {0,1} | 1 | Penalización suave no negativa sqrt(FO^2+ε) en lugar de lineal con signo. |
| `INIT_PIPELINE` | {0,1} | 1 | Activa bloque general de inicialización. |
| `INIT_FROM_ODE` | {0,1} | 0 | Corre una ODE forward nominal y siembra estados (c, cdot) en los FE. |
| `INIT_DUAL_FE` | {0,1} | hereda INIT_PIPELINE | Siembra v, α, λ, y FO_* por FE (heurísticas pFBA). |
| `HOMOTOPY` | {0,1} | 0 | Activa homotopía multi–etapa. |
| `HOM_PHI` | lista | 1e-2,1e-1,1,10 (si HOMOTOPY=1) | Escala de penalización φ por etapa (aplica a FO_L/FO_U/FO_upt). |
| `HOM_W` | lista | 1e-4,1e-5,1e-6,1e-8 | Ridge pFBA w en stationarity por etapa. |
| `HOM_WEIGHTS` | lista int | 1,1,2,4 | Distribución de wall-time relativo por etapa. |
| `W_SSE` | float | 1.0 | Peso SSE en objetivo. |
| `W_PEN` | float | 0.2 (cambiado típicamente a 0.1) | Peso global de la penalización de complementariedad. |
| `WALL_TIME` | seg | 600 | Límite muro total (se reparte si hay homotopía). |

## Flujo recomendado (módulos 0–3)

1. (Mód 0) Preprocesar reducción: ejecutar `pfba_preprocess.jl` ⇒ `reduced_sets.jld2`.
2. (Mód 1) Inicialización primal: `INIT_FROM_ODE=1` para estados coherentes dinámicamente.
3. (Mód 2) Inicialización dual / FO: `INIT_DUAL_FE=1` (por defecto si INIT_PIPELINE=1).
4. (Mód 3) Homotopía: `HOMOTOPY=1` con schedule recomendado (abajo). Se generan métricas por etapa.

## Schedule de homotopía recomendado

| Etapa | φ | w | Peso tiempo | Justificación |
|-------|---|---|-------------|---------------|
| 1 | 1e-2 | 1e-4 | 1 | Penalización suave mínima; favorece ajuste SSE inicial. |
| 2 | 1e-1 | 1e-5 | 1 | Incrementa presión complementaria manteniendo exploración. |
| 3 | 1 | 1e-6 | 2 | Fase de refinamiento SSE con tightening moderado. |
| 4 | 10 | 1e-8 | 4 | Fase de pulido: apretar FO manteniendo estabilidad (evitamos φ=100 por explosión PEN). |

### Racional para pesos 1,1,2,4

Damos más tiempo a etapas donde la no linealidad y rigidez aumentan (φ alto y w bajo) para permitir convergencia parcial antes de forzar límites de tiempo.

### Selección de `W_PEN`

Se observó inflación de PEN y degradación de SSE al subir φ sin ajustar peso. Rango útil: 0.05–0.2. Usar 0.1 como punto de partida. Si `comp_max` baja pero SSE empeora >5–10% respecto a etapa previa, reducir `W_PEN` o frenar en etapa 3.

## Métricas clave

| Métrica | Fuente | Interpretación |
|---------|--------|---------------|
| `SSE` | Expresión JuMP | Ajuste datos (medidos: X,G,F,E). |
| `PEN` | Expresión JuMP | Suma penalizaciones FO ponderadas por φ. |
| `comp_max` | Archivo etapa | Máximo |FO| (incluye uptake); objetivo: decrecer al final (<1e3 en runs reducidos actuales). |
| `stationarity_residual` | Archivo etapa | Máx |LHS stationarity| en reacciones candidatas (proxy optimalidad). Ideal ↓. |
| `OBJ` | Archivo etapa | Objetivo total ponderado. |
| `SSE0`, `PEN0`, `OBJ0` | Baseline | Pre-optimización (para juzgar ganancia relativa). |

### Archivos de métricas

Ejemplo de nombres:

```
results/zenteno_metrics_baseline_20251112-153458.txt
results/zenteno_metrics_hom_s1_20251112-153526.txt
...
```

Campos típicos (etapa):

```
tag=hom_s3
phi=1.0, w=1.0e-6
SSE=2.500000e+05
PEN=4.990000e+06
comp_max=1.880000e+04
stationarity_residual=2.930000e-02
```

### Criterios prácticos de avance / parada

1. Si `stationarity_residual` ≳ 1e-1 y `comp_max` no desciende tras dos etapas → revisar inicialización (quizá activar INIT_FROM_ODE) o bajar φ inicial.
2. Si PEN domina (`W_PEN * PEN >> W_SSE * SSE`) y SSE empeora, reducir `W_PEN` y repetir etapas tardías.
3. Si etapa 3 ofrece mejor SSE y etapa 4 sólo aumenta PEN sin bajar `comp_max` significativamente, considerar detener en 3 para handoff a modelo completo.

## Ejecución rápida (PowerShell)

```
cd "...\julia_deploy"
setx WALL_TIME 80
$env:REDUCED_MODE="1"; $env:INIT_FROM_ODE="1"; $env:INIT_DUAL_FE="1"; `
   $env:HOMOTOPY="1"; $env:HOM_PHI="1e-2,1e-1,1,10"; $env:HOM_W="1e-4,1e-5,1e-6,1e-8"; $env:HOM_WEIGHTS="1,1,2,4"; `
   $env:W_PEN="0.1"; $env:PEN_NONNEG="1"; $env:PEN_REDUCED="1"; `
   julia --project=. .\MPCC_Zenteno.jl
```

## Solución de problemas

| Síntoma | Posible causa | Acción |
|---------|---------------|--------|
| `reduced_sets.jld2` no existe y warnings de reducción | No se corrió `pfba_preprocess.jl` | Ejecutar script; validar permisos de escritura en `results/`. |
| `TIME_LIMIT` en primeras dos etapas con inf_pr alto | φ inicial demasiado agresivo o seeds pobres | Habilitar `INIT_FROM_ODE`; revisar bounds; bajar φ inicial a 1e-3 temporalmente. |
| `PEN` crece varias órdenes en etapa 4, SSE empeora | W_PEN excesivo al tensar φ | Bajar `W_PEN` (p.ej. 0.05) o detener en etapa 3. |
| `stationarity_residual` no baja | w demasiado grande al final | Asegurar etapa final con w ≤ 1e-8; revisar restricciones lambda reducidas. |
| FO_upt todos ~0 en modo reducido cuando esperaba actividad | `REDUCED_DISABLE_UPTAKE=1` o no se incluyeron reacciones de uptake en A/C | Revisar flag y sets generados. |

## Próximos módulos (planeado)

| Módulo | Idea |
|--------|------|
| 4 | Límites de variación temporal en parámetros / smoothness |
| 5 | Refinamiento malla (coarse→fine) usando solución homotopía como warm start |
| 6 | Multistart / clustering inicial de semillas |
| 7 | Reporte activo de sets residualizados (reacciones activas, α saturados) |
| 8 | Ajustes solver adaptativos (cambiar tolerancias tras etapa 2) |
| 9 | Comparador automático de métricas baseline vs nuevas estrategias |

## Resumen rápido de recomendaciones

- Activar reducción (`REDUCED_MODE=1`) para exploración y tuning de schedule; hacer handoff al modelo completo sólo cuando `comp_max` ≲ 1e3 y `stationarity_residual` ≲ 1e-2 (o al menos estable).
- Usar ODE seeding (`INIT_FROM_ODE=1`) cuando los datos distan del start plano; acelera descenso inicial de SSE.
- Mantener `PEN_NONNEG=1` para interpretabilidad (magnitudes directas) salvo análisis comparativos históricos.
- Ajustar `W_PEN` si el ratio (W_PEN*PEN)/(W_SSE*SSE) > 5 temprano; objetivo rango 1–3 en etapas medias.
- Guardar baseline siempre: permite cuantificar ganancias de inicialización antes de modificar schedule.

---
Última actualización: {{AUTO_DOC_FECHA}} (editar manualmente al cambiar parámetros recomendados).


