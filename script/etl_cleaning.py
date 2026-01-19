import pandas as pd
import os

# ==============================================================================
# Nettoyage du dataset des résultat de 2022 au premier tour => T1
# ==============================================================================

t1 = pd.read_excel("data_raw/resultats-par-niveau-dpt-t1-france-entiere.xlsx")
colonnes_fixes = t1.columns[:17].tolist()
candidates = t1.columns[17:]
blocs = [candidates[i:i+6] for i in range(0, len(candidates), 6)]
dfs = []
for bloc in blocs:
    df_bloc = t1[colonnes_fixes + list(bloc)].copy()
    df_bloc.columns = colonnes_fixes + ["Sexe", "Nom", "Prénom", "Voix", "% Voix/Ins", "% Voix/Exp"]
    dfs.append(df_bloc)
t1_clean = pd.concat(dfs, ignore_index=True)
t1_clean = t1_clean.rename(columns={
    "Code du département": "code_insee",
    "Libellé du département": "departement",
    "Prénom": "prenom",
    "Nom": "nom",
    "Voix": "voix",
    "% Voix/Exp": "voix_exp"
})
t1_clean["tour"] = 1
t1_clean.to_csv("data_clean/elections_t1.csv", index=False)
print(" T1 nettoyé => présent dans data_clean/elections_t1.csv")


# ==============================================================================
# Nettoyage du dataset des résultat de 2022 au deuxieme tour => T2
# ==============================================================================

t2 = pd.read_excel("data_raw/resultats-par-niveau-dpt-t2-france-entiere.xlsx")
colonnes_fixes = t2.columns[:17].tolist()
candidates = t2.columns[17:]
blocs = [candidates[i:i+6] for i in range(0, len(candidates), 6)]
dfs = []
for bloc in blocs:
    df_bloc = t2[colonnes_fixes + list(bloc)].copy()
    df_bloc.columns = (
        colonnes_fixes +
        ["Sexe", "Nom", "Prénom", "Voix", "% Voix/Ins", "% Voix/Exp"]
    )
    dfs.append(df_bloc)
t2_clean = pd.concat(dfs, ignore_index=True)
t2_clean = t2_clean.rename(columns={
    "Code du département": "code_insee",
    "Libellé du département": "departement",
    "Prénom": "prenom",
    "Nom": "nom",
    "Voix": "voix",
    "% Voix/Exp": "voix_exp"
})
t2_clean["tour"] = 2
t2_clean.to_csv("data_clean/elections_t2.csv", index=False)
print("T2 nettoyé => présent dans data_clean/elections_t2.csv")

# ==============================================================================
# Nettoyage du dataset emploie => t3
# ==============================================================================

t3 = pd.read_excel("data_raw/sl_etc_2025T3.xls", header=3)
id_vars = ["Code", "Libellé"]
value_vars = [col for col in t3.columns if "_" in col]
tcl_long = t3.melt(
    id_vars=id_vars,
    value_vars=value_vars,
    var_name="periode",
    value_name="taux_chomage"
)
tcl_long["trim"] = tcl_long["periode"].str.split("_").str[0]
tcl_long["annee"] = tcl_long["periode"].str.split("_").str[1].astype(int)
tcl_long = tcl_long.drop(columns=["periode"])
tcl_long = tcl_long.rename(columns={
    "Code": "code_insee",
    "Libellé": "departement"
})
tcl_2022_T3 = tcl_long[
    (tcl_long["annee"] == 2022) &
    (tcl_long["trim"] == "T4")
]
tcl_2022_T3.to_csv("data_clean/chomage_2022_T3.csv", index=False)
print("T3 nettoyé => présent dans data_clean/chomage_2022_T3.csv")

# ==============================================================================
# Fusion T1 + T2 + T3 → Table Finale
# ==============================================================================

t1_clean = pd.read_csv("data_clean/elections_t1.csv")
t2_clean = pd.read_csv("data_clean/elections_t2.csv")
elections = pd.concat([t1_clean, t2_clean], ignore_index=True)
chomage = pd.read_csv("data_clean/chomage_2022_T3.csv")
tf = elections.merge(
    chomage[["code_insee", "taux_chomage"]],
    on="code_insee",
    how="left"
)
tf.to_csv("data_clean/table_finale.csv", index=False)
print("table_final crée => présent dans data_clean/table_finale.csv")


# ==============================================================================
# Utilisation du dataset "table_final" et filtrage par Hauts-de-France
# ==============================================================================
tf = pd.read_csv("data_clean/table_finale.csv")
hdf_codes = ["02", "59", "60", "62", "80"]
tf_hdf = tf[tf["code_insee"].astype(str).isin(hdf_codes)]
tf_hdf.to_csv("data_clean/table_finale_HDF.csv", index=False)
print("table_final_HDF créer => présent dans data_clean/table_finale_HDF.csv")

# ==============================================================================
# Utilisation du dataset "Table_final_HDF" pour ajouter les familles politiques
# ==============================================================================

tf_hdf = pd.read_csv("data_clean/table_finale_HDF.csv")
def classer_famille(nom):
    nom = nom.upper()
    if nom in ["ARTHAUD", "ROUSSEL", "POUTOU", "MÉLENCHON", "MELENCHON", "HIDALGO", "JADOT"]:
        return "gauche"
    if nom in ["MACRON"]:
        return "centre"
    if nom in ["PÉCRESSE", "PECRESSE", "DUPONT-AIGNAN", "LASSALLE"]:
        return "droite"
    if nom in ["LE PEN", "ZEMMOUR"]:
        return "extreme_droite"
    return "autre"
tf_hdf["famille_politique"] = tf_hdf["nom"].apply(classer_famille)
tf_hdf.to_csv("data_clean/table_finale_HDF_famille.csv", index=False)

print("table_final_HDF_famille créer => présent dans data_clean/table_finale_HDF_famille.csv")
# ==============================================================================
# Utilisation du dataset "Table_final" pour ajouter les familles politiques
# ==============================================================================

tf_full = pd.read_csv("data_clean/table_finale.csv")
def classer_famille(nom):
    nom = nom.upper()
    if nom in ["ARTHAUD", "ROUSSEL", "POUTOU", "MÉLENCHON", "MELENCHON", "HIDALGO", "JADOT"]:
        return "gauche"
    if nom in ["MACRON"]:
        return "centre"
    if nom in ["PÉCRESSE", "PECRESSE", "DUPONT-AIGNAN", "LASSALLE"]:
        return "droite"
    if nom in ["LE PEN", "ZEMMOUR"]:
        return "extreme_droite"
    return "autre"
tf_full["famille_politique"] = tf_full["nom"].apply(classer_famille)
tf_full.to_csv("data_clean/table_finale_famille.csv", index=False)
print("table_final_famille créer => présent dans data_clean/table_finale_famille.csv")
