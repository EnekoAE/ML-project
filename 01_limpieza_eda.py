import ast
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Crear carpeta para guardar las gráficas del documento
os.makedirs("figures", exist_ok=True)

# 2. Cargar el dataset bruto de la carpeta dataset
df = pd.read_csv("dataset/dataset_euskadi_raw.csv")
print(f"Dataset original: {df.shape[0]} filas x {df.shape[1]} columnas")

# 3. Eliminar columnas que están 100% vacías
cols_vacias = df.columns[df.isnull().all()].tolist()
df = df.drop(columns=cols_vacias)
print(f"Eliminadas {len(cols_vacias)} columnas 100% vacías.")

# 4. Arreglar tildes/encoding en los nombres de provincia y municipio
for col in ["neighbourhood_cleansed", "neighbourhood_group_cleansed"]:
    df[col] = (
        df[col]
        .str.replace("GuipÃºzcoa", "Gipuzkoa", regex=False)
        .str.replace("Ã\x81lava", "Álava", regex=False)
        .str.replace("Ã¡", "á", regex=False)
        .str.replace("Ã©", "é", regex=False)
        .str.replace("Ã­", "í", regex=False)
        .str.replace("Ã³", "ó", regex=False)
        .str.replace("Ãº", "ú", regex=False)
    )

# 5. Limpiar el precio ("$237.00" -> 237.0) y sacar nº de amenities y baños
df["price_num"] = (
    df["price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)
df["num_amenities"] = df["amenities"].apply(
    lambda x: len(ast.literal_eval(x)) if pd.notnull(x) and x.startswith("[") else 0
)
df["bathrooms_num"] = df["bathrooms_text"].str.extract(r"(\d+\.?\d*)").astype(float)

# 6. Quedarnos solo con las 46 columnas útiles (sin URLs, IDs ni texto libre)
columnas_utiles = [
    "host_is_superhost",
    "price_num",
    "neighbourhood_group_cleansed",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms_num",
    "bedrooms",
    "beds",
    "num_amenities",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "estimated_occupancy_l365d",
    "estimated_revenue_l365d",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "hosts_time_as_host_years",
    "host_has_profile_pic",
    "host_identity_verified",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "pob_2025",
    "crec_pob_2016_2025_pct",
    "renta_disponible_2024",
    "peso_renta_inmob_pct",
    "n_vut",
    "plazas_reate_por_1000_hab",
    "n_aloj_tradicionales",
    "plazas_trad_por_1000_hab",
    "categoria_reate",
    "nucleo_urbano",
]

df_clean = df[columnas_utiles].copy()
df_clean.to_csv("dataset/dataset_euskadi_clean.csv", index=False)
print(f"Dataset limpio guardado en dataset/dataset_euskadi_clean.csv: {df_clean.shape}")

# 7. Generar las 3 gráficas exploratorias iniciales
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sns.countplot(
    data=df_clean,
    x="neighbourhood_group_cleansed",
    order=["Vizcaya", "Gipuzkoa", "Álava"],
    palette="Blues_d",
    ax=axes[0],
)
axes[0].set_title("Listings by Province")
axes[0].set_xlabel("Province")
axes[0].set_ylabel("Count")

sns.countplot(data=df_clean, x="host_is_superhost", palette="Set2", ax=axes[1])
axes[1].set_title("Superhost Distribution (Target)")
axes[1].set_xlabel("Is Superhost (t / f)")
axes[1].set_ylabel("Count")

sns.histplot(
    df_clean[df_clean["price_num"] <= 1000]["price_num"].dropna(),
    bins=40,
    kde=True,
    color="#1B365D",
    ax=axes[2],
)
axes[2].set_title("Nightly Price Distribution (<= 1000)")
axes[2].set_xlabel("Price per Night")
axes[2].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("figures/eda_overview.png", dpi=300)
print("Gráfica guardada en figures/eda_overview.png")