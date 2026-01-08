# Configuración de Variables de Entorno para MPCC_Zenteno_stripping.jl

## Variables principales para generación de datos sintéticos MPCC

### 1. Generar cache de datos MPCC (solo una vez)
```powershell
$env:EXPORT_MPCC_SYNTH_DATA = "1"
$env:MPCC_SYNTH_DATA_PATH = "mpcc_synth_data.jld2"  # Opcional, este es el default
julia --project=. .\MPCC_Zenteno_stripping.jl
```
**Resultado:** Resuelve MPCC con parámetros "reales" y guarda los estados en cache JLD2.

---

### 2. Usar datos MPCC cacheados (modo rápido, sin re-solver)
```powershell
$env:USE_MPCC_SYNTH_DATA = "1"
julia --project=. .\MPCC_Zenteno_stripping.jl
```
**Resultado:** 
- Carga datos experimentales desde cache MPCC
- Genera curva pre-optimización (línea discontinua) desde splines del cache
- Resuelve MPCC con parámetros a calibrar
- Genera curva post-optimización (línea continua) desde splines de la solución

---

### 3. Configuración completa recomendada (workflow típico)

#### Primera vez (generar datos):
```powershell
# Generar datos MPCC sintéticos con parámetros "reales"
$env:EXPORT_MPCC_SYNTH_DATA = "1"
$env:MPCC_SYNTH_DATA_PATH = ".\mpcc_synth_data.jld2"
$env:EXPERIMENT = "MPCC_GEN_DATA_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
julia --project=. .\MPCC_Zenteno_stripping.jl
```

#### Corridas posteriores (calibración):
```powershell
# Usar datos MPCC cacheados
$env:USE_MPCC_SYNTH_DATA = "1"
$env:MPCC_SYNTH_DATA_PATH = ".\mpcc_synth_data.jld2"
$env:EXPERIMENT = "CALIB_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$env:EXPORT_PLOT_CSV = "0"  # Cambiar a "1" si quieres CSVs exportados
julia --project=. .\MPCC_Zenteno_stripping.jl
```

---

## Variables de entorno disponibles (referencia completa)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `USE_MPCC_SYNTH_DATA` | `"0"` | Cargar datos sintéticos desde cache MPCC (JLD2) |
| `MPCC_SYNTH_DATA_PATH` | `"mpcc_synth_data.jld2"` | Ruta del archivo cache |
| `MPCC_SYNTH_FORCE` | `"0"` | Forzar regeneración del cache (ignorar existente) |
| `EXPORT_MPCC_SYNTH_DATA` | `"0"` | Guardar datos MPCC en cache tras resolver |
| `EXPORT_MPCC_PRE_CURVE` | `"0"` | *(Reservado)* Exportar curva pre por separado |
| `EXPORT_PLOT_CSV` | `"0"` | Exportar datos de gráficos como CSV |
| `USE_WARM_START` | `"1"` | Usar semilla de optimización previa |
| `EXPERIMENT` | `"default"` | Token para organizar outputs en subdirectorio |
| `DIAG_SIMPLE` | `"0"` | Modo diagnóstico sin complementariedad |
| `REDUCED_MODE` | `"0"` | Usar modelo GEM reducido |
| `T_CONST` | `"293.15"` | Temperatura base [K] |
| `STRIP_LIQ_VOL` | `"100.0"` | Volumen líquido para stripping [L] |
| `STRIP_AROMA_INIT` | `"0.0"` | Concentración inicial de aroma [g/L] |

---

## Interpretación del gráfico resultante

### Con `USE_MPCC_SYNTH_DATA=1`:

- **Línea discontinua (dashdot):** Spline de MPCC con parámetros "reales" (θ_final del reporte)
  - Labels: "MPCC pre X", "MPCC pre N", etc.
  
- **Línea continua (sólida):** Spline de MPCC optimizado (parámetros calibrados)
  - Labels: "MPCC post X", "MPCC post N", etc.
  
- **Puntos morados (diamantes):** Puntos de colocación MPCC optimizados
  - Labels: "Colocación MPCC X", etc.
  
- **Puntos círculos (color estado):** Datos "experimentales" MPCC con parámetros reales
  - Labels: "Datos MPCC real G", "Datos MPCC real F", "Datos MPCC real E"
  - Solo para estados medidos (G, F, E)

### Sin cache (fallback ODE):
- Datos y curvas generados con modelo ODE de Zenteno
- Labels permanecen como "ODE pre/post" (modo legacy)

---

## Troubleshooting

### Error: "Cache MPCC sin clave 'data'"
**Causa:** Archivo cache corrupto o incompleto  
**Solución:** Re-generar con `MPCC_SYNTH_FORCE=1` y `EXPORT_MPCC_SYNTH_DATA=1`

### Warning: "USE_MPCC_SYNTH_DATA=1 pero no existe el archivo cache"
**Causa:** Primera corrida sin haber generado el cache  
**Solución:** Primero ejecutar paso 1 (generar cache)

### Curvas muy diferentes entre pre y post
**Normal:** Indica que la calibración ajustó significativamente los parámetros  
**Verificar:** Revisar `theta_final_log` en el reporte de salida

### Datos experimentales no se ven
**Causa:** Estados no están en `MEAS_STATES = (3, 4, 5)` (G, F, E)  
**Solución:** Estados X, N, O2 no se muestran como "datos" por defecto

---

## Ejemplo de workflow completo

```powershell
# 1. Generar datos MPCC "reales" una sola vez
cd "C:\...\Yeast_83\estima"
$env:EXPORT_MPCC_SYNTH_DATA = "1"
$env:EXPERIMENT = "GEN_SYNTH_DATA"
julia --project=. .\MPCC_Zenteno_stripping.jl

# 2. Verificar que se creó mpcc_synth_data.jld2
ls .\mpcc_synth_data.jld2

# 3. Corridas de calibración (reutilizando datos)
$env:USE_MPCC_SYNTH_DATA = "1"
$env:EXPORT_MPCC_SYNTH_DATA = "0"  # Ya no guardar
$env:EXPERIMENT = "CALIB_RUN_01"
julia --project=. .\MPCC_Zenteno_stripping.jl

# 4. Nueva calibración con diferentes settings
$env:EXPERIMENT = "CALIB_RUN_02"
julia --project=. .\MPCC_Zenteno_stripping.jl
```

Cada corrida generará PNG en `plots/EXPERIMENT_NAME/` con timestamp único.
