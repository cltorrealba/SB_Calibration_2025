# run_mpcc_gendata.ps1
# Genera datos sintéticos MPCC CON COMPLEMENTARIEDAD (parámetros "reales" θ_final FIJADOS)

Write-Host "=== Generando datos MPCC sintéticos (CON complementariedad) ===" -ForegroundColor Cyan
Write-Host "Parámetros 'reales' FIJADOS (theta_final del reporte):" -ForegroundColor Yellow
Write-Host "  mu0  = 0.141665" -ForegroundColor Gray
Write-Host "  Yeg  = 0.14133" -ForegroundColor Gray
Write-Host "  Yef  = 0.96932" -ForegroundColor Gray
Write-Host "  Yxn  = 9.80576" -ForegroundColor Gray
Write-Host "  mrate= 1.0001" -ForegroundColor Gray
Write-Host ""
Write-Host "NOTA: Complementariedad ACTIVA (DIAG_SIMPLE=0)" -ForegroundColor Yellow
Write-Host "NOTA: Parámetros FIJADOS (FIX_THETA_TO_FINAL=1)" -ForegroundColor Yellow
Write-Host "Esto genera datos 'reales' pero toma 1-2 horas." -ForegroundColor Yellow
Write-Host ""

$env:EXPORT_MPCC_SYNTH_DATA = "1"
$env:MPCC_SYNTH_DATA_PATH = ".\mpcc_synth_data.jld2"
$env:EXPERIMENT = "GEN_SYNTH_DATA_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$env:USE_WARM_START = "0"
$env:DIAG_SIMPLE = "0"
$env:ESTIMATE_PARAMS = "false"
$env:FIX_THETA_TO_FINAL = "1"
$env:IPOPT_LINEAR_SOLVER = "mumps"

Write-Host "Ejecutando Julia con IPOPT (paciencia, 1-2 horas)..." -ForegroundColor Green
$startTime = Get-Date

julia --project=. .\MPCC_Zenteno_stripping.jl

$endTime = Get-Date
$duration = $endTime - $startTime

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Generación completada ===" -ForegroundColor Green
    Write-Host "Tiempo total: $($duration.TotalMinutes -as [int]) minutos" -ForegroundColor Cyan
    
    if (Test-Path ".\mpcc_synth_data.jld2") {
        $fileSize = (Get-Item ".\mpcc_synth_data.jld2").Length / 1MB
        Write-Host "Cache creado: mpcc_synth_data.jld2 ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Próximos pasos:" -ForegroundColor Yellow
        Write-Host "  1. Inspeccionar datos: .\run_mpcc_calib.ps1" -ForegroundColor Gray
        Write-Host "  2. Calibrar vs datos: .\run_mpcc_calib.ps1" -ForegroundColor Gray
    } else {
        Write-Host "ADVERTENCIA: No se encontró mpcc_synth_data.jld2" -ForegroundColor Red
    }
} else {
    Write-Host "ERROR: Julia finalizó con código $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Revisa logs en ./plots/" -ForegroundColor Red
}
