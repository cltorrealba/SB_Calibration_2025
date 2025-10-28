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
