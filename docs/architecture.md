# Arquitectura propuesta — SB_Calibration_2025

Este documento describe una arquitectura mínima y práctica para reorganizar el repositorio
de calibración de modelos de fermentación (SB_Calibration_2025). El objetivo es separar
responsabilidades, facilitar tests, y permitir ejecuciones reproducibles.

## Resumen

- Propósito: preparar datos experimentales (2024/2025), construir matrices de calibración,
  ajustar el modelo dinámico (Zenteno), y validar/guardar artefactos.
- Problemas actuales: scripts monolíticos, I/O disperso, reglas de parsing duplicadas,
  y falta de tests/CI.

## Estructura propuesta (src/package)

Organizar el código como un paquete Python bajo `src/sb_calibration/` con módulos claros:

- src/sb_calibration/
  - data/
    - loader.py            # Cargar ensayos (2024/2025) y normalizar columnas
    - schema.py            # Nombres canónicos de columnas / constantes
    - persistence.py       # Lectura / escritura de artefactos (parquet, csv)
  - preprocess/
    - preprocess_2024.py   # Pipeline SW_Preprocess_data refactorizado (now: sw_preprocess.py / sw_full.py)
    - preprocess_2025.py   # Pipeline Calibration_data_preprocess refactorizado (now: calibration_preprocess.py)
    - sugar_density.py     # Ajuste densidad -> azúcar (fit, predict, export)
    - util.py              # helpers reutilizables (fechas, interpolación)
  - model/
    - zenteno.py           # zenteno_model, jacobiano, simuladores (stiff/RK4)
    - profiles.py          # Construcción de perfiles T/pulsos
  - calibration/
    - objective.py         # SSE, normalizaciones y balanceos
    - optimize.py          # multistart + DE, early stop, checkpoints
    - pulses.py            # Inferencia de pulsos desde química
  - metadata/
    - build.py             # lógica de `metadata.py`
    - partition.py         # lógica de `data_partition.py`
  - cli/
    - calibrate_cli.py     # Entrypoint para calibración (splits, 2024 builder, cache, plots)
    - preprocess_cli.py    # Entrypoint para preprocesado (2024/2025)
  - viz/
    - plots.py             # Todas las funciones de visualización

Esta separación facilita:
- tests unitarios por módulo,
- reutilización del simulador desde notebooks o servicios,
- caché/persistencia de artefactos intermedios.

## Flujo de datos (simplificado)

1. Raw Excel / hojas (`Datos Experimentales/`, `Procesos_I+D_2025_3.xlsx`) → `data.loader`
2. Preprocessing 2024/2025 → `preprocess` produce matrices por ensayo (normalizadas)
3. `sugar_density.fit()` (opcional) → modelo densidad→azúcar y artefactos en `sugar_density_out/`
4. `model` proporciona `simulate(p, x0, temp_profile, pulses, t_eval)`
5. `calibration.optimize` usa `objective.sse_for_experiments_real` + `model.simulate` para estimar parámetros
6. Resultados y artefactos (parquet, initial_conditions.csv, pbest_checkpoint.npz) → `mats/`

## Configuración y ejecución reproducible

- Usar un archivo de configuración (YAML/JSON) para rutas, seeds y parámetros de optimización.
- Entrypoints/CLI pequeños en `src/sb_calibration/cli/` leen la configuración y ejecutan
  el pipeline completo o pasos parciales (preprocess, fit sugar, calibrate, validate).

## Testing y CI

- Tests unitarios (pytest) por:
  - parsing y normalización de fechas
  - construcción de matrices (casos con/sin temperatura, densidad)
  - simulador: sanity-check (no NaN, monotonicidades esperadas)
  - función objetivo: SSE sobre dataset sintético
- CI (GitHub Actions): lint (ruff/black), pytest (fast tests), build check.

## Notas de rendimiento

- Paralelizar evaluaciones costosas (DE) y/o emplear caching para simulaciones repetidas.
- Mantener checkpointing periódico (pbest_checkpoint.npz) y logs estructurados.

## Siguientes pasos recomendados

1. Crear el skeleton del paquete (`src/..`) y mover `modelo_dinamico_sim.py` como primer paso.
2. Añadir tests de humo y CI minimal.
3. Refactorizar preprocessores en módulos pequeños y testables.

Migration progress (2025-10-02)
--------------------------------
- `calibration_preprocess.py` implemented under `src/sb_calibration/preprocess/` with extraction,
  isotonic correction, smoothing and matrix building.
- `sw_preprocess.py` and `sw_full.py` added as compatibility and SW-specific helpers.
- Tests added for smoke/compatibility flows and the repo CI updated to install package in editable mode.

Migration note (2025-10-02)
---------------------------------
Se han movido los scripts monolíticos originales a la carpeta `legacy/` para mantener una copia
histórica y permitir avanzar en la refactorización del código en `src/`. Los archivos movidos incluyen
los preprocessores y el script de calibración original. Esto facilita la creación de módulos
limpios en `src/sb_calibration/` sin perder trazabilidad.

Continuous Integration (CI) — breve explicación
---------------------------------
CI (Integración continua) es el proceso mediante el cual se ejecutan pruebas automáticas y checks
cada vez que se hacen cambios en el repositorio (push / PR). En este proyecto añadimos un workflow
de CI que ejecuta `pytest` para validar que los cambios no rompan la base de código. Recomendamos
extender el pipeline para incluir linters, formateo y comprobaciones de seguridad.


Calibración refactorizada (2025-10-06)
--------------------------------------
Estado actual de la migración del orquestador y simulación:

- Simulador (stiff por tramos con pulsos): `model.zenteno.simulate_on_grid`
  - solve_ivp (Radau/BDF), con soporte de jacobiano analítico (`zenteno_jacobian` + `J_SPARSE`),
    jacobiano numérico (complex-step/forward-diff) o sin jacobiano.
  - Temperatura entrada como segmentos o DataFrame con `time_h` + `Temperature_C`.
  - Pulsos de N aplicados instantáneamente al final de cada tramo, como en el legacy.

- Objetivo: `calibration.objective.sse_for_experiments_real`
  - SSE normalizado por std global por variable (X,N,G,F,E), con `N_SCALE=1e-3` para mg/L→g/L.
  - Interpolación de simulación a tiempos de medición.

- Optimizador: `calibration.optimize.calibrate_full`
  - Modos `multistart` y `de` (DE + pulido local), reparametrización en z-space.
  - Early stopping por evaluaciones y starts, checkpoint incremental (`pbest_checkpoint.npz`) y reanudación.

- CLI: `cli/calibrate_cli.py`
  - Modo (`--mode`), `--n-starts`, `--local-maxiter`, tolerancias (`--rtol`, `--atol-*`),
    solver (`--method`), jacobiano (`--jacobian`), `--file`/`--temps-dir`/`--assays`, `--exclude`, `--use-smoothed-biomass`.
  - Splits train/valid (`--split`, `--split-file`). Si hay 24xxx, se construyen on-the-fly desde `Datos Experimentales/Data <ID>.xlsx` (módulo `preprocess.sw_2024`) con cache opcional (`--no-cache-2024`).
  - Pulsos desde química (`--chem-file`) y pesos por variable (`--w-x ...`).
  - Prebuild de 24xxx (`--prebuild-2024`, `--prebuild-split`, `--prebuild-only`).
  - Plots por ensayo (`--plot`, `--plots-dir`) y `--verbose` para trazabilidad.

Paridad con legacy (Calibración global / jac_ana)
-------------------------------------------------
Replicado:
- Integración stiff con jacobiano y sparsidad, por tramos con pulsos.
- Normalizaciones (std global, N en g/L), SSE sobre (X,N,G,F,E) cuando estén presentes.
- Modos de optimización (multistart/DE) con reparam log y early-stop.
- Checkpoint incremental y reanudación.

Completado recientemente:
- Construcción automática de pulsos desde planilla química vía `calibration.pulses.build_pulses_from_chem` e integración en la CLI con `--chem-file`.
- Exposición de pesos por variable (`WEIGHTS`) como flags (`--w-x --w-n --w-g --w-f --w-e`) y propagación hasta el objetivo.

Completado también:
- Visualización del ajuste por ensayo (plots) en `viz/plots.py` y opción `--plot` en la CLI.

Riesgos/Notas:
- Los defaults de `n_starts`, `maxiter` y tolerancias están acotados para tests rápidos. Para calibraciones reales conviene incrementarlos.
- El jacobiano numérico es útil para validación; en producción usar el analítico para mejor estabilidad y performance.

