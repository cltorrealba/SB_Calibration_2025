# run_mpcc_calib.ps1
# Ejecuta calibración MPCC usando datos sintéticos cacheados

param(
    [string]$ExperimentName = "CALIB_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [switch]$ExportCSV = $false
)

Write-Host "=== Calibración MPCC ===" -ForegroundColor Cyan
Write-Host "Experimento: $ExperimentName" -ForegroundColor Yellow

# Verificar que existe el cache
if (-not (Test-Path ".\mpcc_synth_data.jld2")) {
    Write-Host "ERROR: No se encontró mpcc_synth_data.jld2" -ForegroundColor Red
    Write-Host "Primero ejecuta: .\run_mpcc_gendata.ps1" -ForegroundColor Yellow
    exit 1
}

$env:USE_MPCC_SYNTH_DATA = "1"
$env:MPCC_SYNTH_DATA_PATH = ".\mpcc_synth_data.jld2"
$env:EXPORT_MPCC_SYNTH_DATA = "0"  # No sobreescribir los datos reales
$env:EXPERIMENT = $ExperimentName
$env:USE_WARM_START = "1"

if ($ExportCSV) {
    $env:EXPORT_PLOT_CSV = "1"
    Write-Host "Exportación de CSV activada" -ForegroundColor Gray
} else {
    $env:EXPORT_PLOT_CSV = "0"
}

Write-Host "Cargando datos desde cache..." -ForegroundColor Green
Write-Host "Ejecutando Julia..." -ForegroundColor Green
julia --project=. .\MPCC_Zenteno_stripping.jl

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Calibración completada ===" -ForegroundColor Green
    $plotsDir = ".\plots\$ExperimentName"
    if (Test-Path $plotsDir) {
        Write-Host "Resultados en: $plotsDir" -ForegroundColor Cyan
        Get-ChildItem $plotsDir -Filter "*.png" | ForEach-Object {
            Write-Host "  - $($_.Name)" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "ERROR: Julia finalizó con código $LASTEXITCODE" -ForegroundColor Red
}
