# SB_Calibration_2025

Calibración de un modelo ODE de fermentación (Zenteno) con datos 2024/2025. Incluye preprocesado, simulación stiff con jacobiano analítico, función objetivo con pesos, optimización multistart/DE, plots y soporte de splits train/valid.

Quickstart (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Prebuild opcional de ensayos 24xxx desde "Datos Experimentales"
python -m sb_calibration.cli.calibrate_cli --prebuild-2024 --prebuild-split all --temps-dir "Datos Experimentales" --verbose

# Calibración típica (usa split train)
python -m sb_calibration.cli.calibrate_cli `
	--file "Procesos_I+D_2025_3.xlsx" `
	--temps-dir "Datos Experimentales" `
	--split train `
	--method Radau `
	--jacobian analytic `
	--mode multistart `
	--n-starts 12 `
	--local-maxiter 300 `
	--rtol 1e-6 `
	--plot --plots-dir mats/plots `
	--out mats/pbest_checkpoint.npz `
	--verbose
```

Documentación

- Guía de uso: `docs/usage.md`
- Arquitectura: `docs/architecture.md`
- Diccionario de códigos: `docs/code_dictionary.md`

Notas

- Scripts legacy permanecen en `legacy/` como referencia.
- La CLI construye on-the-fly matrices 24xxx si aparecen en el split y puede cachearlas en `mats/` (desactivar con `--no-cache-2024`).

## Julia MPCC (Ipopt) – Experimentos

Ubicación: `MPCC/De Oliveira 2023/Yeast_83/estima/julia_deploy`

- Ejecutable: `experiment_pipeline.jl`
- Modos soportados: `baseline`, `seed`, `seed_run`, `seed180`, `seed180_bv`, `seed180_cf`, `seed_reduced`, `seed_frozen`, `seed_run_frozen`, `bv_trial`, `compare`, `init_only`.

### Solver por defecto y Pardiso/HSL (opt‑in)

- Por defecto: `linear_solver=mumps` para no consumir licencias de Pardiso.
- Para habilitar Pardiso por corrida, exporta:

```powershell
$env:IPOPT_LINEAR_SOLVER="pardiso"
# Recomendado (Pardiso Panua):
$env:PARDISO_NUM_THREADS="2"
$env:PARDISO_MATCHING="complete+2x2"
$env:PARDISO_ORDER="metis"
$env:PARDISO_MSG_LVL="0"
# Rutas (si usas Ipopt/Pardiso de Panua):
$env:PANUA_IPOPT_DIR="C:\ruta\panua-ipopt-20240228-win"
$env:IPOPT_PARDISO_DLL_DIR="C:\ruta\panua-pardiso-20240630-win\lib"
$env:PANUA_LIC_PATH="C:\ruta\panua-licenses"
```

Notas:
- El pipeline silencia el banner de licencia con `PARDISOLICMESSAGE=1` y fija `OMP_NUM_THREADS`/`MKL_NUM_THREADS` si `PARDISO_NUM_THREADS` está definido.
- Si `IPOPT_LINEAR_SOLVER` no está seteado, se usa MUMPS.

Para habilitar HSL (MA77/MA57/MA86/MA97), necesitas la versión **completa** de `HSL_jll.jl` (no la dummy):

#### Paso 1: Descargar HSL_jll completo

1. Visita https://licences.stfc.ac.uk/product/libhsl
2. Descarga `HSL_jll.jl.v2024.11.28.zip` (o versión más reciente)
3. Extrae en una carpeta, por ejemplo: `C:\HSL_jll`

**Nota macOS:** Quitar quarantine antes de extraer:
```bash
xattr -d com.apple.quarantine HSL_jll.jl.v2024.11.28.zip
```

#### Paso 2: Instalar HSL_jll en modo desarrollo

```powershell
# Desde Julia REPL (una sola vez)
julia> ]
pkg> dev C:\HSL_jll  # o la ruta donde extrajiste HSL_jll
```

#### Paso 3: Usar solver HSL

Luego, por corrida, exporta:

```powershell
$env:IPOPT_LINEAR_SOLVER="ma77"   # o "ma57" | "ma86" | "ma97"
julia --project=. .\MPCC_Zenteno_stripping.jl
```

Notas HSL:
- **Versión dummy vs completa:** El `HSL_jll` del registro público de Julia es una versión "dummy" sin solvers. Debes descargar e instalar la versión completa desde el sitio de HSL.
- **Verificar instalación:** Ejecuta `julia --project=. .\check_hsl.jl` para verificar si tienes la versión funcional.
- Los scripts automáticamente:
  - Detectan HSL_jll y cargan OpenBLAS32 (LP64 BLAS requerido)
  - Configuran `hsllib` apuntando a `HSL_jll.libhsl_path`
  - Si HSL_jll no está funcional, muestran error con instrucciones
- Compatible con Julia ≥ 1.9 (recomendado para libblastrampoline).
- Más info: https://github.com/JuliaSmoothOptimizers/HSL.jl

### Comandos típicos (PowerShell)

1) Baseline 360s (sin homotopía, MUMPS por defecto):

```powershell
Set-Location -Path "...\MPCC\De Oliveira 2023\Yeast_83\estima\julia_deploy"
$env:EXPERIMENT="mi_experimento"
$env:ACTIVE_REPORT="1"; $env:SKIP_PLOTS="1"; $env:WALL_TIME="360"; $env:NFE="12"
julia --project=. .\experiment_pipeline.jl baseline
```

2) Inicialización (init_only) 360s para generar warm‑start (opcionalmente con Pardiso):

```powershell
# (opcional) activar Pardiso por esta corrida
$env:IPOPT_LINEAR_SOLVER="pardiso"; $env:PARDISO_NUM_THREADS="2"; $env:PARDISO_MATCHING="complete+2x2"; $env:PARDISO_ORDER="metis"; $env:PARDISO_MSG_LVL="0"
$env:INIT_FROM_ODE="1"; $env:INIT_DUAL_FE="1"; $env:BV_ON="0"
julia --project=. .\experiment_pipeline.jl init_only
```

Salidas clave en `results/<EXPERIMENT>/`:
- `zenteno_handoff_full_checkpoint.jld2` (warm‑start completo)
- `zenteno_seed_checkpoint.jld2` (copia del anterior para consumo de `seed_run`)
- `zenteno_estimation_report_*.txt`, `zenteno_metrics_init_only_*.txt`

3) Corrida seeded 360s desde el checkpoint:

```powershell
julia --project=. .\experiment_pipeline.jl seed_run
```

### Cuándo conviene usar la semilla (warm‑start)

- Mismo modelo/estructura y `nfe`: reduce penalizaciones y complementarias, acelera y estabiliza la convergencia.
- Multistart: mejor tasa de éxito por intento; combina con variación de inicios de parámetros.
- Coarse→Fine: soportado por `seed180_cf` (mapeo de malla integrado).
- Cambios fuertes (estructura, BV, datos muy distintos): considera regenerar la semilla.
- Aislar solver vs init: puedes correr MUMPS con `INIT_FROM_CHECKPOINT=1` y el mismo `CHECKPOINT_PATH`.

### Recomendaciones Pardiso (si se usa)

- `PARDISO_NUM_THREADS=2` suele ser estable y con buen rendimiento.
- `pardiso_matching_strategy=complete+2x2`, `pardiso_order=metis`, `pardiso_msglvl=0`.
- Mantén Ipopt primero en `PATH`; añade la carpeta de DLL de Pardiso al final.
