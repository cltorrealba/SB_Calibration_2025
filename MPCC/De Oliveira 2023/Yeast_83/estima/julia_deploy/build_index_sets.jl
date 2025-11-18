#!/usr/bin/env julia
# build_index_sets.jl
#
# Construye:
#  - Mapas ID -> índice para reacciones y metabolitos (según yeastGEM.xlsx)
#  - Índices de metabolitos redox (NAD/NADH/NADP/NADPH)
#  - Conjunto de reacciones redox (columnas de S con esos mets)
#  - Índices de biomasa y mantenimiento
#  - Conjunto de reacciones de N (desde Modelo_Acoplado FVA Yeast8 N.py)
#
# Salida: index_sets.jld2

using XLSX
using DelimitedFiles
using JLD2

# ───────────────────────────────────────────────────────────────────────────────
# Rutas base (AJUSTA ESTAS 3 SI ES NECESARIO)
# ───────────────────────────────────────────────────────────────────────────────

# Directorio donde están yeastGEM.xlsx y S.csv
const ESTIMA_DIR = @__DIR__      # o pon ruta absoluta si prefieres

const MODEL_XLSX = joinpath(ESTIMA_DIR, "yeastGEM.xlsx")
const S_CSV      = joinpath(ESTIMA_DIR, "S.csv")
const OUT_JLD2   = joinpath(ESTIMA_DIR, "index_sets.jld2")

println("====================================================================")
println("[INDEX] Generando conjuntos de índices para yeastGEM")
println("  ESTIMA_DIR = $ESTIMA_DIR")
println("  MODEL_XLSX = $MODEL_XLSX")
println("  S_CSV      = $S_CSV")
println("====================================================================")

# ───────────────────────────────────────────────────────────────────────────────
# 1) Leer Excel (RXNS y METS) y construir mapas ID -> índice
# ───────────────────────────────────────────────────────────────────────────────

@assert isfile(MODEL_XLSX) "No se encontró el archivo yeastGEM.xlsx en $MODEL_XLSX"
@assert isfile(S_CSV)      "No se encontró el archivo S.csv en $S_CSV"

println("[INDEX] Leyendo yeastGEM.xlsx…")
xlsx = XLSX.readxlsx(MODEL_XLSX)

rxns_sheet = xlsx["RXNS"]
mets_sheet = xlsx["METS"]

function read_column(sheet::XLSX.Worksheet, name::String)
    header_row = vec(sheet[1, :])
    col_idx = findfirst(==(name), header_row)
    col_idx === nothing && error("No se encontró columna '$name' en hoja $(sheet.name)")
    col_idx isa CartesianIndex && (col_idx = col_idx[2])

    dim = XLSX.get_dimension(sheet)
    dim === nothing && error("La hoja $(sheet.name) no tiene dimensión declarada")
    last_row = XLSX.row_number(dim.stop)

    return [string(sheet[r, col_idx]) for r in 2:last_row]
end


rxn_ids   = read_column(rxns_sheet, "ID")
rxn_names = read_column(rxns_sheet, "NAME")
met_ids   = read_column(mets_sheet, "ID")

rxn_id_to_idx = Dict{String,Int}()
for (j, rid) in enumerate(rxn_ids)
    rid_clean = strip(rid)
    if !isempty(rid_clean)
        rxn_id_to_idx[rid_clean] = j  # j es la columna j de S
    end
end

met_id_to_idx = Dict{String,Int}()
for (i, mid) in enumerate(met_ids)
    mid_clean = strip(mid)
    if !isempty(mid_clean)
        met_id_to_idx[mid_clean] = i  # i es la fila i de S
    end
end

println("[INDEX] Reacciones total (RXNS):   ", length(rxn_id_to_idx))
println("[INDEX] Metabolitos total (METS): ", length(met_id_to_idx))

# ───────────────────────────────────────────────────────────────────────────────
# 2) Metabolitos redox (NAD/NADH/NADP/NADPH) → índices de filas en S
# ───────────────────────────────────────────────────────────────────────────────

nad_met_ids = [
    "NAD[c]","NAD[er]","NAD[m]","NAD[n]","NAD[p]",
    "NADH[c]","NADH[er]","NADH[m]","NADH[p]",
    "NADP(+)[c]","NADP(+)[er]","NADP(+)[m]","NADP(+)[p]",
    "NADPH[c]","NADPH[er]","NADPH[m]","NADPH[p]"
]

nad_met_idx = Int[]
println("\n[INDEX] Buscando metabolitos redox (NAD/NADH/NADP/NADPH)…")
for mid in nad_met_ids
    if haskey(met_id_to_idx, mid)
        idx = met_id_to_idx[mid]
        push!(nad_met_idx, idx)
        println("  ✔ $mid  -> fila S = $idx")
    else
        @warn "[INDEX] Metabolito redox no encontrado en METS" mid
    end
end
println("[INDEX] Metabolitos redox encontrados: ", length(nad_met_idx))

# ───────────────────────────────────────────────────────────────────────────────
# 3) Leer S.csv y detectar reacciones redox (columnas con coeficiente ≠ 0 en NAD*)
# ───────────────────────────────────────────────────────────────────────────────

println("\n[INDEX] Leyendo S.csv…")
S = readdlm(S_CSV, ',', Float64)  # nm x nr
nm, nr = size(S)
println("[INDEX] Dimensiones de S: nm (filas) = $nm, nr (columnas) = $nr")

@assert nm ≥ maximum(nad_met_idx) "Algún índice de metabolito redox excede las filas de S"

redox_rxn_idx = Int[]
for j in 1:nr
    # ¿Algún coeficiente ≠ 0 en las filas NAD/NADH/NADP/NADPH?
    if any(abs(S[i, j]) > 0 for i in nad_met_idx)
        push!(redox_rxn_idx, j)
    end
end

println("[INDEX] Reacciones redox detectadas (tocan NAD*/NADH*/NADP*/NADPH*): ",
        length(redox_rxn_idx))

# Muestra resumen de las primeras 10 para inspección visual
println("\n[CHECK] Primeras 10 reacciones redox:")
for (k, j) in enumerate(redox_rxn_idx[1:min(end, 10)])
    rid = j <= length(rxn_ids) ? rxn_ids[j] : "N/A"
    rname = j <= length(rxn_names) ? rxn_names[j] : "N/A"
    println("  [$k] col=$j  ID=$rid  NAME=$rname")
end

# ───────────────────────────────────────────────────────────────────────────────
# 4) Índices de biomasa y mantenimiento (por NAME)
# ───────────────────────────────────────────────────────────────────────────────

function find_rxn_by_name_substr(substr::String)
    idxs = [j for (j, nm) in enumerate(rxn_names) if occursin(substr, nm)]
    return idxs
end

biomass_idxs = find_rxn_by_name_substr("biomass pseudoreaction")
maint_idxs   = find_rxn_by_name_substr("non-growth associated maintenance reaction")

if length(biomass_idxs) != 1
    @warn "[INDEX] Biomass pseudoreaction no fue única" biomass_idxs
end
if length(maint_idxs) != 1
    @warn "[INDEX] Maintenance reaction no fue única" maint_idxs
end

biomass_idx = isempty(biomass_idxs) ? 0 : biomass_idxs[1]
maint_idx   = isempty(maint_idxs)   ? 0 : maint_idxs[1]

println("\n[INDEX] Biomass pseudoreaction → col S = $biomass_idx",
        biomass_idx > 0 ? " (ID=$(rxn_ids[biomass_idx]))" : "")
println("[INDEX] Non-growth maintenance → col S = $maint_idx",
        maint_idx > 0 ? " (ID=$(rxn_ids[maint_idx]))" : "")

# ───────────────────────────────────────────────────────────────────────────────
# 5) Reacciones de Nitrógeno (desde Modelo_Acoplado FVA Yeast8 N.py)
# ───────────────────────────────────────────────────────────────────────────────
# Extraídas automáticamente de ese archivo (bloque de consumo de N / YAN):

nitrogen_rxn_ids = [
    "r_1654",  # Amonia
    "r_1891",  # Gln
    "r_1879",  # Arg
    "r_1899",  # Leu
    "r_1897",  # Ile
    "r_1911",  # Thr
    "r_1914",  # Val
    "r_1873",  # Ala
    "r_1880",  # Asp
    "r_1883",  # Cys
    "r_1889",  # Glu
    "r_1893",  # His
    "r_1900",  # Lys
    "r_1902",  # Met
    "r_1903",  # Phe
    "r_1906",  # Ser
    "r_1912",  # Trp
    "r_1913",  # Tyr
    "r_1810",  # Gly
    # También incluidos en ese mismo bloque, aunque no todos sean N-puros,
    # pero útiles para bounds dinámicos acoplados:
    "r_1714",  # Glucose uptake (en ese script)
    "r_1709",  # Fructose uptake
    "r_1761",  # Ethanol
    "r_1672",  # CO2
]

nitrogen_rxn_idx = Int[]
println("\n[INDEX] Buscando reacciones de N (y asociadas) en RXNS…")
for rid in nitrogen_rxn_ids
    if haskey(rxn_id_to_idx, rid)
        j = rxn_id_to_idx[rid]
        push!(nitrogen_rxn_idx, j)
        rname = rxn_names[j]
        println("  ✔ $rid -> col S = $j  NAME=$rname")
    else
        @warn "[INDEX] RxN Nitrógeno no encontrada en RXNS" rid
    end
end

println("[INDEX] Total reacciones N marcadas: ", length(nitrogen_rxn_idx))

# ───────────────────────────────────────────────────────────────────────────────
# 6) Guardar todo en JLD2 para usarlo en el MPCC
# ───────────────────────────────────────────────────────────────────────────────

println("\n[INDEX] Guardando conjuntos en: $OUT_JLD2")

@save OUT_JLD2 nad_met_idx redox_rxn_idx biomass_idx maint_idx nitrogen_rxn_idx rxn_id_to_idx met_id_to_idx rxn_ids rxn_names met_ids

println("[INDEX] Listo. Revisa la salida arriba para verificar que:")
println("  - Biomass y maintenance apuntan a las reacciones correctas.")
println("  - Las primeras reacciones redox tienen sentido fisiológico.")
println("  - Las reacciones de N muestran los aminoácidos y NH4 esperados.")
println("====================================================================")
