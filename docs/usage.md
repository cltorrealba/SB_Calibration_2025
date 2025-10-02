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


