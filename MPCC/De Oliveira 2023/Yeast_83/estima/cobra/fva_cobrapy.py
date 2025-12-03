# mini_reduce_for_MPCC.py
#
# Punto de partida para reducción del GEM usando COBRApy
# alineado con tu MPCC (Zenteno + stripping).

import sys
from pathlib import Path

import cobra
from cobra.flux_analysis import flux_variability_analysis


def load_cobra_model(path_str: str):
    """Load a COBRA model picking the right reader by file extension."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    ext = path.suffix.lower()
    if ext in {".xml", ".sbml"}:
        return cobra.io.read_sbml_model(path)
    if ext == ".json":
        return cobra.io.load_json_model(path)
    if ext == ".mat":
        return cobra.io.load_matlab_model(path)

    raise ValueError(
        f"Unsupported model format '{ext}'. "
        "Convert the Excel workbook (yeastGEM.xlsx) to SBML/JSON/MAT first "
        "and point model_path to that file."
    )

# 1. Cargar el modelo GEM (Yeast8 extendido)
#    Reemplaza por la ruta real a tu modelo (SBML .xml/.sbml, .json o .mat)
model_path = "C:\\Users\\ctorrealba\\OneDrive - Viña Concha y Toro S.A\\Documentos\\Proyectos I+D\\PI-4497\\Resultados\\2025\\SB_Calibration_2025\\MPCC\\De Oliveira 2023\\Yeast_83\\estima\\cobra\\yeast-GEM.xml"
try:
    model = load_cobra_model(model_path)
except Exception as err:
    sys.exit(
        f"[ERROR] No se pudo leer el modelo '{model_path}'. "
        f"{err}\n"
        "Descarga/convierte el modelo a SBML (.xml/.sbml), JSON o MAT y actualiza model_path."
    )

print(f"Modelo cargado: {len(model.metabolites)} mets, {len(model.reactions)} rxns")

# 2. Definir reacciones relevantes vistas desde el MPCC
#    *** IMPORTANTE ***: reemplaza los IDs placeholder por los de tu modelo.

CORE_GROUPS = {
    # a) Uptake y secreción principales (G, F, E, N, CO2, ácidos, glicerol)
    "exchange": [
        "r_1714",        # glucosa
        "r_1709",        # fructosa
        "r_1654",        # amonio
        "r_1761",        # etanol
        "r_1808",        # glicerol
        "r_1634",        # acetato
        "r_2056",        # succinato
        "r_1546",        # lactato
        "r_1672",        # CO2
        "r_1634",        # acetato
    ],

    # b) Biomasa y mantenimiento
    "biomass_energy": [
        "r_4041",          # o la que uses como 'obj'
        "r_4046",          # mantenimiento no asociado a crecimiento
    ],

    # c) Ruta glucosa/fructosa → piruvato → etanol / TCA / glicerol
    "central_carbon": [
        "r_0534",  # Hexokinasa (HXK1/HXK2)
        "r_0533",  # Hexokinasa (HXK1/HXK2)
        "r_0467",  # Glucosa-6-fosfato isomerasa (phosphoglucose isomerase)
        "r_0886",  # Fosfofructoquinasa (PFK1/PFK2)
        "r_0450",  # Fructosa-bisfosfato aldolasa (aldolase)
        "r_1054",  # Triosa-fosfato isomerasa (TPI1)
        "r_0486",  # Gliceraldehído-3P deshidrogenasa (GAPDH)
        "r_0892",  # Fosfoglicerato quinasa (PGK)
        "r_0893",  # Fosfoglicerato mutasa (PGM)
        "r_0366",  # Enolasa (ENO1/ENO2)
        "r_0962",  # Piruvato quinasa (PYK1/2)
        "r_0959",  # Piruvato descarboxilasa (PDC1/5/6)
        "r_0960",  # Piruvato descarboxilasa (PDC1/5/6) acetoina
        "r_0163",  # Alcohol deshidrogenasa (ADH1/2/3/5)
        "r_0164",  # Alcohol deshidrogenasa (ADH1/2/3/5)
    ],

        # d) Ramas redox / subproductos
    "central_carbon": [
        "r_0490",  # Glicerol-3-fosfato deshidrogenasa (GPD1/GPD2)
        "r_0491",  # Glicerol-3-fosfato deshidrogenasa (GPD1/GPD2)
        "r_0492",  # Glicerol-3-fosfato deshidrogenasa (GPD1/GPD2)
        "r_0489",  # Glicerol-3-fosfatasa (GPP1/GPP2)
        "r_0534",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0166",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0167",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0168",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0169",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0170",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0171",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0172",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0173",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0174",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0175",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0176",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0177",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0178",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0179",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0180",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0181",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0182",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0183",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0184",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0185",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0186",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0187",  # Acetaldehído deshidrogenasa (ALD isoformas relevantes en citosol)
        "r_0112",  # Acetil-CoA sintetasa (ACS) – para reasimilación de acetato, si está en el modelo
        "r_0454",  # Succinate dehydrogenase / fumarate reductase (SDH/FRD, la isoforma que permita fumarato ↔ succinato)
        "r_0455",  # Succinate dehydrogenase / fumarate reductase (SDH/FRD, la isoforma que permita fumarato ↔ succinato)
        "r_1000",  # Succinate dehydrogenase / fumarate reductase (SDH/FRD, la isoforma que permita fumarato ↔ succinato)
        "r_1021",  # Succinate dehydrogenase
        "r_0451",  # Fumarasa (FUM1)
        "r_0452",  # Fumarasa (FUM1)
        "r_0713",  # Malate dehydrogenase (MDH1/2)
        "r_0714",  # Malate dehydrogenase (MDH1/2)
        "r_0715",  # Lactato deshidrogenasa (LDH)
        
    ],

    # d) Asimilación de nitrógeno (YAN)
    "nitrogen": [
        "R_NH4t",     # uptake NH4+
        "R_GLn_synth",
        "R_GLt_synth",
        # añade las rutas mínimas que conectan NH4/AA a biomasa
    ],

    # e) Aroma (ethyl acetate y otros, si los vas a usar)
    "aroma": [
        "r_1765",        # ethyl acetate (ID real de tu reacción de ethyl acetate)
    ],
}

# 3. Construir el conjunto de reacciones 'core' y verificar que existan
core_rxn_ids = set()
for group, rxns in CORE_GROUPS.items():
    for rxn_id in rxns:
        if rxn_id not in model.reactions:
            print(f"[WARN] Reacción {rxn_id} (grupo {group}) no está en el modelo. Revísala.")
        else:
            core_rxn_ids.add(rxn_id)

print(f"Total reacciones 'core' encontradas en el modelo: {len(core_rxn_ids)}")

# 4. Fijar una condición de medio simple (ejemplo)
#    Aquí puedes usar los mismos bounds que usas en el MPCC (lb/ub para glu, fru, N, etc.)
medium = model.medium
if "EX_glc__D_e" in medium:
    medium["EX_glc__D_e"] = 10.0  # mmol/gDW/h max uptake
if "EX_fru_e" in medium:
    medium["EX_fru_e"] = 10.0
if "EX_nh4_e" in medium:
    medium["EX_nh4_e"] = 5.0
model.medium = medium

# 5. Ejecutar una FBA base para chequear viabilidad
solution = model.optimize()
print(f"FBA base: status={solution.status}, growth={solution.objective_value}")

# 6. FVA sobre las reacciones core (para valorar qué tanto se “mueven”)
core_rxns = [model.reactions.get_by_id(r) for r in core_rxn_ids]
fva_res = flux_variability_analysis(model, reaction_list=core_rxns, fraction_of_optimum=0.9)

print("\nFVA para reacciones core (fracción 0.9 del óptimo):")
for rxn_id, row in fva_res.iterrows():
    print(f"{rxn_id:25s}  min={row['minimum']:8.3f}  max={row['maximum']:8.3f}")

# 7. (Opcional) Punto de partida para construir un modelo reducido:
#    - Mantener core_rxn_ids
#    - Mantener cualquier reacción esencial para factibilidad (por ejemplo, las que
#      tengan FVA no trivial o que conecten metabolitos internos necesarios)
#
# Aquí solo dejamos un placeholder:
#
# from cobra.manipulation.delete import remove_reactions
#
# to_remove = [r for r in model.reactions
#              if r.id not in core_rxn_ids]
# reduced_model = model.copy()
# remove_reactions(reduced_model, to_remove, remove_orphans=True)
#
# cobra.io.write_sbml_model(reduced_model, "yeast8_reduced_for_MPCC.xml")
