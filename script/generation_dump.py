import pandas as pd
import sqlite3
import os

# ============================================================
# Configurtion du dump
# ============================================================

CSV_FILE = "data_clean/table_finale_HDF_famille.csv"

TABLE_NAME = "table_propre"            
DB_FILE = "database.db"             
DUMP_FILE = "dump_table_final_GODEFROY_Meline.sql"              

# ============================================================
# Chargement du CSV
# ============================================================

print(f"Chargement du fichier CSV : {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

print(f"{len(df)} lignes chargées.")
print("Colonnes détectées :", list(df.columns))

# ============================================================
# Creation de la base sqlite et insertion des données
# ============================================================

print(f"Creation de la base sqlite : {DB_FILE}")
conn = sqlite3.connect(DB_FILE)
df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
print(f"Table '{TABLE_NAME}' créée avec succès dans {DB_FILE}.")

# ============================================================
# Export du dump sql
# ============================================================

print(f"Génération du dump sql : {DUMP_FILE}")
with open(DUMP_FILE, "w", encoding="utf-8") as f:
    for line in conn.iterdump():
        f.write(f"{line}\n")
conn.close()
print("Dump sql généré avec succès.")
print("Fichiers créés :")
print(f" - Base sqlite : {DB_FILE}")
print(f" - Dump sql    : {DUMP_FILE}")
