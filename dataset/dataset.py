import pandas as pd

# cargar el dataset
df_raw = pd.read_csv(
    "libreria/listings.csv",
    encoding="utf-8",
    low_memory=False
)

print("Contar filas e id unicos")
print(f"Total filas: {df_raw.shape[0]} | Total columnas: {df_raw.shape[1]}")
print(f"IDs únicos: {df_raw['id'].nunique()}")

# limpieza de oprecio y valores no nulos
df_raw["price_num"] = pd.to_numeric(
    df_raw["price"].astype(str).str.replace(r"[$,]", "", regex=True),
    errors="coerce"
)

print("\nCalidad de dataset")
print(f"Con precio valido (>0): {(df_raw['price_num'] > 0).mean() * 100:.2f}% ({(df_raw['price_num'] > 0).sum()} anuncios)")
print(f"Con licencia informada: {df_raw['license'].notna().mean() * 100:.2f}% ({df_raw['license'].notna().sum()} anuncios)")
print(f"Con coordenadas completas: {(df_raw['latitude'].notna() & df_raw['longitude'].notna()).mean() * 100:.2f}%")

#contar por capitales
print("\nAnuncios por capital")
muni = df_raw["neighbourhood_cleansed"].fillna("")
for capital, patron in [("Bilbao", "bilbao"), ("Donostia", "donostia"), ("Vitoria-Gasteiz", "vitoria")]:
    subset = df_raw[muni.str.lower().str.contains(patron)]
    con_precio = (subset["price_num"] > 0).sum()
    print(f"- {capital}: {len(subset)} anuncios totales | {con_precio} con precio válido")

print("\nTop 10 municipios con más anuncios:")
print(df_raw["neighbourhood_cleansed"].value_counts().head(10))

# 4. Revisión de desbalanceo para variables candidatas a clasificación
print("\n=== 4. CANDIDATAS A CLASIFICACIÓN ===")
print("\nDistribución de room_type:")
print(df_raw["room_type"].value_counts(normalize=True) * 100)

df_raw["es_profesional"] = df_raw["calculated_host_listings_count"] > 1
print("\nDistribución anfitrión profesional (>1 anuncio):")
print(df_raw["es_profesional"].value_counts(normalize=True) * 100)