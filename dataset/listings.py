from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "libreria"
OUT_DIR = Path(__file__).resolve().parent


def fix_mojibake(text: str) -> str:
    if pd.isna(text):
        return ""
    s = str(text)
    try:
        return s.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def normalize_name(name: str) -> str:
    s = fix_mojibake(name).lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[/\-\(\)\.,]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_mun_lookup(df_mun: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in df_mun.iterrows():
        code = str(row["cod_mun"]).zfill(5)
        mun = fix_mojibake(str(row["municipio"]))
        lookup[normalize_name(mun)] = code
        if "/" in mun:
            for part in mun.split("/"):
                p_norm = normalize_name(part)
                if p_norm and p_norm not in lookup:
                    lookup[p_norm] = code
    extra_aliases = {
        "sopelana": "48085",
        "karrantza": "48022",
        "arrankudiaga zollo": "48009",
        "erriberabeitia": "01047",
        "turtzioz": "48087",
        "trucios": "48087",
        "arrazua ubarrundia": "01008",
        "oyon": "01043",
        "moreda de alava": "01039",
        "legutiano": "01058",
        "mungai": "48069",
        "san millan": "01053",
        "yecora": "01060",
        "bidegoian": "20024",
        "etxebarri anteiglesia de san esteban etxebarri doneztebeko elizatea": "48029",
        "garay": "48039",
        "donostia san sebastian": "20069",
        "abadino": "48001",
        "abanto y ciervana abanto zierbena": "48002",
        "ereno": "48033",
        "lantaron": "01902",
        "valdegovia": "01055",
        "alegria dulantzi": "01001",
        "onati": "20059",
        "urduna orduna": "48074",
        "valle de trapaga trapagaran": "48080",
        "manaria": "48059",
        "lagran": "01030",
        "banos de ebro": "01011",
        "iruna oka": "01901",
        "arminon": "01006",
        "menaka": "48064",
    }
    lookup.update(extra_aliases)
    return lookup


def parse_currency(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[\$,€\s]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_bool_tf(series: pd.Series) -> pd.Series:
    return series.map({"t": 1, "f": 0, True: 1, False: 0}).fillna(0).astype(int)


def load_listings(listings_path: Path, df_mun: pd.DataFrame, df_reate: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(listings_path, low_memory=False)
    mun_lookup = build_mun_lookup(df_mun)

    mun_col = "neighbourhood_cleansed" if "neighbourhood_cleansed" in df.columns else "neighbourhood"
    df[mun_col] = df[mun_col].apply(fix_mojibake)
    df["cod_mun"] = df[mun_col].apply(normalize_name).map(mun_lookup)

    sin_cruce = df[df["cod_mun"].isna()][mun_col].dropna().unique().tolist()
    if sin_cruce:
        print("Municipios en listings sin cruce:", sin_cruce)
    else:
        print("Cruce municipal en listings: 100% completado sin nulos.")

    df["precio_noche"] = parse_currency(df["price"])
    if "price_quote_price_per_night" in df.columns:
        df["precio_noche"] = df["precio_noche"].fillna(parse_currency(df["price_quote_price_per_night"]))

    df["es_superhost"] = parse_bool_tf(df["host_is_superhost"])
    df["reserva_inmediata"] = parse_bool_tf(df["instant_bookable"])

    reate_regs = set(df_reate["n_registro"].dropna().astype(str).str.strip().str.upper())
    df["licencia_limpia"] = df["license"].fillna("").astype(str).str.strip().str.upper()
    df["tiene_licencia_declarada"] = (df["licencia_limpia"] != "").astype(int)
    df["en_reate"] = df["licencia_limpia"].isin(reate_regs).astype(int)

    df = df.rename(
        columns={
            "neighbourhood_group_cleansed": "territorio_airbnb",
            mun_col: "municipio_airbnb",
            "accommodates": "plazas_anuncio",
            "bathrooms": "banos",
            "bedrooms": "dormitorios",
            "beds": "camas",
            "minimum_nights": "noches_minimas",
            "availability_30": "disponibilidad_30d",
            "availability_365": "disponibilidad_365d",
            "number_of_reviews": "n_resenas",
            "number_of_reviews_ltm": "n_resenas_12m",
            "estimated_occupancy_l365d": "ocupacion_estimada_dias",
            "estimated_revenue_l365d": "ingresos_estimados_anuales",
            "review_scores_rating": "valoracion_general",
            "review_scores_location": "valoracion_ubicacion",
            "review_scores_value": "valoracion_calidad_precio",
            "calculated_host_listings_count": "anuncios_del_anfitrion",
            "reviews_per_month": "resenas_por_mes",
        }
    )

    keep_cols = [
        "id",
        "cod_mun",
        "territorio_airbnb",
        "municipio_airbnb",
        "latitude",
        "longitude",
        "property_type",
        "room_type",
        "plazas_anuncio",
        "banos",
        "dormitorios",
        "camas",
        "precio_noche",
        "noches_minimas",
        "disponibilidad_30d",
        "disponibilidad_365d",
        "n_resenas",
        "n_resenas_12m",
        "resenas_por_mes",
        "ocupacion_estimada_dias",
        "ingresos_estimados_anuales",
        "valoracion_general",
        "valoracion_ubicacion",
        "valoracion_calidad_precio",
        "host_id",
        "es_superhost",
        "anuncios_del_anfitrion",
        "reserva_inmediata",
        "licencia_limpia",
        "tiene_licencia_declarada",
        "en_reate",
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    return df[existing_cols].copy()


def build_final_datasets(data_dir: Path, out_dir: Path):
    df_mun = pd.read_csv(out_dir / "dataset_municipal.csv", dtype={"cod_mun": str, "cod_prov": str})
    df_reate = pd.read_csv(out_dir / "reate_unificado.csv", dtype={"cod_mun": str})
    df_listings = load_listings(data_dir / "listings.csv", df_mun, df_reate)

    valid_listings = df_listings.dropna(subset=["cod_mun"]).copy()
    valid_listings["es_vivienda_entera"] = (valid_listings["room_type"] == "Entire home/apt").astype(int)
    valid_listings["es_habitacion"] = (
        valid_listings["room_type"].isin(["Private room", "Shared room"]).astype(int)
    )
    valid_listings["es_multianfitrion"] = (valid_listings["anuncios_del_anfitrion"] > 1).astype(int)

    agg_air = valid_listings.groupby("cod_mun", as_index=False).agg(
        n_anuncios_airbnb=("id", "count"),
        plazas_airbnb=("plazas_anuncio", "sum"),
        n_viviendas_enteras_airbnb=("es_vivienda_entera", "sum"),
        n_habitaciones_airbnb=("es_habitacion", "sum"),
        precio_noche_medio=("precio_noche", "mean"),
        precio_noche_mediano=("precio_noche", "median"),
        ocupacion_estimada_media=("ocupacion_estimada_dias", "mean"),
        ingresos_anuales_medios=("ingresos_estimados_anuales", "mean"),
        valoracion_media=("valoracion_general", "mean"),
        pct_superhosts=("es_superhost", lambda s: s.mean() * 100),
        pct_multianfitrion=("es_multianfitrion", lambda s: s.mean() * 100),
        pct_licencia_declarada=("tiene_licencia_declarada", lambda s: s.mean() * 100),
        pct_en_reate=("en_reate", lambda s: s.mean() * 100),
    )

    df_final = df_mun.merge(agg_air, on="cod_mun", how="left")

    zero_air_cols = [
        "n_anuncios_airbnb",
        "plazas_airbnb",
        "n_viviendas_enteras_airbnb",
        "n_habitaciones_airbnb",
    ]
    for col in zero_air_cols:
        df_final[col] = df_final[col].fillna(0)

    df_final["anuncios_airbnb_por_1000_hab"] = (df_final["n_anuncios_airbnb"] / df_final["pob_2025"]) * 1000
    df_final["ratio_airbnb_sobre_reate"] = np.where(
        df_final["total_alojamientos_reate"] > 0,
        df_final["n_anuncios_airbnb"] / df_final["total_alojamientos_reate"],
        np.nan,
    )
    df_final["indice_precio_turistico_vs_renta"] = (
        (df_final["precio_noche_mediano"] * 30) / (df_final["renta_disponible_2024"] / 12)
    ) * 100

    mun_ctx = df_final[["cod_mun", "pob_2025", "renta_disponible_2024", "plazas_reate_por_1000_hab"]]
    df_listings_enriched = df_listings.merge(mun_ctx, on="cod_mun", how="left")

    df_listings_enriched.to_csv(out_dir / "listings_limpio.csv", index=False, encoding="utf-8-sig")
    df_final.to_csv(out_dir / "dataset_final_municipal.csv", index=False, encoding="utf-8-sig")
    return df_final, df_listings_enriched


if __name__ == "__main__":
    df_final, df_listings = build_final_datasets(DATA_DIR, OUT_DIR)
    print("dataset_final_municipal.csv generado:", df_final.shape)
    print("listings_limpio.csv generado:", df_listings.shape)