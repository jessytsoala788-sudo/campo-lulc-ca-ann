
"""
Calcule la superficie reelle (en ha) de chaque classe dans les rasters
simules 2035 et 2045, et dans le raster de validation (2025 simule).
A executer une fois les rasters produits par le pipeline principal.
"""
import numpy as np
import rasterio

CLASSES = {
    1: "Foret dense",
    2: "Foret degradee",
    3: "Agriculture",
    4: "Mangrove",
    5: "Eau",
    6: "Sol nu/Bati",
}

HA_PAR_PIXEL = 0.09  # 30m x 30m = 900 m2 = 0.09 ha

def superficies_par_classe(chemin_raster):
    with rasterio.open(chemin_raster) as src:
        arr = src.read(1)
    resultats = {}
    for code, nom in CLASSES.items():
        n_pixels = int(np.sum(arr == code))
        resultats[nom] = n_pixels * HA_PAR_PIXEL
    return resultats

if __name__ == "__main__":
    fichiers = {
        "2025 (validation, simule depuis 2015)": "C:\\mnt\\user-data\\outputs\\resultats_python\\LULC_2025_simule_validation.tif",
        "2035": "C:\\mnt\\user-data\\outputs\\resultats_python\\LULC_simule_2035.tif",
        "2045": "C:\\mnt\\user-data\\outputs\\resultats_python\\LULC_simule_2045.tif",
    }

    print(f"{'Classe':18s}", end="")
    for label in fichiers:
        print(f"{label:>35s}", end="")
    print()

    toutes_superficies = {label: superficies_par_classe(chemin) for label, chemin in fichiers.items()}

    for nom in CLASSES.values():
        print(f"{nom:18s}", end="")
        for label in fichiers:
            print(f"{toutes_superficies[label][nom]:35,.1f}", end="")
        print()