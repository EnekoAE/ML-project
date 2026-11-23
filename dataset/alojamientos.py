from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "libreria"
OUT_DIR = Path(__file__).resolve().parent


def load_reate(viv_path: Path, hab_path: Path) -> pd.DataFrame:
    df_viv = pd.read_json(viv_path)
    df_hab = pd.read_json(hab_path)

    df_viv["categoria_reate"] = "VUT"
    df_hab["categoria_reate"] = "Habitacion"

    for col in ["Nresidentesvivienda", "Ndormitoriosresidentes", "Ndormitoriosusuarios"]:
        if col not in df_viv.columns:
            df_viv[col] = np.nan

    cols = [
        "Nregistro",
        "FechainscripcionREATE",
        "categoria_reate",
        "Tipoalojamiento",
        "Modalidad",
        "Nombrecomercial",
        "Direccion",
        "Codigopostal",
        "Localidad",
        "Codigomunicipio",
        "Municipio",
        "Codigoprovincia",
        "Provincia",
        "Capacidad",
        "Nbanos",
        "Superficietotalviviendasm2",
        "Situadaennucleorural",
        "Situadaennucleourbano",
        "Nresidentesvivienda",
        "Ndormitoriosresidentes",
        "Ndormitoriosusuarios",
    ]

    df = pd.concat([df_viv[cols], df_hab[cols]], ignore_index=True)

    df["cod_mun"] = (
        df["Codigoprovincia"].astype(str).str.strip().str.zfill(2)
        + df["Codigomunicipio"].astype(str).str.strip().str.zfill(3)
    )
    df["fecha_inscripcion"] = pd.to_datetime(
        df["FechainscripcionREATE"], format="%d/%m/%Y", errors="coerce"
    )
    df["anio_inscripcion"] = df["fecha_inscripcion"].dt.year
    df["Codigopostal"] = df["Codigopostal"].astype(str).str.strip().str.zfill(5)

    rename_map = {
        "Nregistro": "n_registro",
        "Tipoalojamiento": "tipo_alojamiento",
        "Modalidad": "modalidad",
        "Nombrecomercial": "nombre_comercial",
        "Direccion": "direccion",
        "Codigopostal": "codigo_postal",
        "Localidad": "localidad",
        "Municipio": "municipio_reate",
        "Provincia": "provincia_reate",
        "Capacidad": "capacidad",
        "Nbanos": "n_banos",
        "Superficietotalviviendasm2": "superficie_m2",
        "Situadaennucleorural": "nucleo_rural",
        "Situadaennucleourbano": "nucleo_urbano",
        "Nresidentesvivienda": "n_residentes_vivienda",
        "Ndormitoriosresidentes": "n_dormitorios_residentes",
        "Ndormitoriosusuarios": "n_dormitorios_usuarios",
    }
    df = df.rename(columns=rename_map)

    final_cols = [
        "n_registro",
        "categoria_reate",
        "tipo_alojamiento",
        "modalidad",
        "fecha_inscripcion",
        "anio_inscripcion",
        "cod_mun",
        "provincia_reate",
        "municipio_reate",
        "localidad",
        "codigo_postal",
        "direccion",
        "nombre_comercial",
        "capacidad",
        "n_banos",
        "superficie_m2",
        "nucleo_rural",
        "nucleo_urbano",
        "n_residentes_vivienda",
        "n_dormitorios_residentes",
        "n_dormitorios_usuarios",
    ]
    return df[final_cols]


def load_traditional(aloj_path: Path) -> pd.DataFrame:
    with open(aloj_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["cod_mun"] = (
        df["territorycode"].astype(str).str.strip().str.zfill(2)
        + df["municipalitycode"].astype(str).str.strip().str.zfill(3)
    )
    df["capacidad_num"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(0)
    return df.groupby("cod_mun", as_index=False).agg(
        n_aloj_tradicionales=("documentName", "count"),
        plazas_aloj_tradicionales=("capacidad_num", "sum"),
        n_hoteles_pensiones=("lodgingType", lambda s: s.isin(["Hoteles", "Pensiones", "Hotel-Apartamento"]).sum()),
        n_rurales_agroturismos=("lodgingType", lambda s: s.isin(["Casas Rurales", "Agroturismos"]).sum()),
        n_apartamentos_reg=("lodgingType", lambda s: (s == "Apartamentos").sum()),
    )


def build_datasets(data_dir: Path, out_dir: Path):
    df_base = pd.read_csv(out_dir / "base_municipal.csv", dtype={"cod_mun": str, "cod_prov": str})
    df_reate = load_reate(data_dir / "viviendas.json", data_dir / "habitaciones.json")
    df_trad = load_traditional(data_dir / "alojamientos.json")

    vut = df_reate[df_reate["categoria_reate"] == "VUT"]
    hab = df_reate[df_reate["categoria_reate"] == "Habitacion"]

    agg_vut = vut.groupby("cod_mun", as_index=False).agg(
        n_vut=("categoria_reate", "count"),
        plazas_vut=("capacidad", "sum"),
        sup_media_vut_m2=("superficie_m2", "mean"),
        vut_rural=("nucleo_rural", lambda s: (s == "Sí").sum()),
        vut_urbano=("nucleo_urbano", lambda s: (s == "Sí").sum()),
        vut_alta_pre2020=("anio_inscripcion", lambda s: (s <= 2019).sum()),
        vut_alta_2020_2023=("anio_inscripcion", lambda s: ((s >= 2020) & (s <= 2023)).sum()),
        vut_alta_2024_2026=("anio_inscripcion", lambda s: (s >= 2024).sum()),
    )

    agg_hab = hab.groupby("cod_mun", as_index=False).agg(
        n_habitaciones=("categoria_reate", "count"),
        plazas_habitaciones=("capacidad", "sum"),
        sup_media_hab_m2=("superficie_m2", "mean"),
        hab_rural=("nucleo_rural", lambda s: (s == "Sí").sum()),
        hab_urbano=("nucleo_urbano", lambda s: (s == "Sí").sum()),
        hab_alta_pre2020=("anio_inscripcion", lambda s: (s <= 2019).sum()),
        hab_alta_2020_2023=("anio_inscripcion", lambda s: ((s >= 2020) & (s <= 2023)).sum()),
        hab_alta_2024_2026=("anio_inscripcion", lambda s: (s >= 2024).sum()),
    )

    df_mun = (
        df_base.merge(agg_vut, on="cod_mun", how="left")
        .merge(agg_hab, on="cod_mun", how="left")
        .merge(df_trad, on="cod_mun", how="left")
    )

    zero_cols = [
        "n_vut",
        "plazas_vut",
        "vut_rural",
        "vut_urbano",
        "vut_alta_pre2020",
        "vut_alta_2020_2023",
        "vut_alta_2024_2026",
        "n_habitaciones",
        "plazas_habitaciones",
        "hab_rural",
        "hab_urbano",
        "hab_alta_pre2020",
        "hab_alta_2020_2023",
        "hab_alta_2024_2026",
        "n_aloj_tradicionales",
        "plazas_aloj_tradicionales",
        "n_hoteles_pensiones",
        "n_rurales_agroturismos",
        "n_apartamentos_reg",
    ]
    for col in zero_cols:
        df_mun[col] = df_mun[col].fillna(0)

    df_mun["total_alojamientos_reate"] = df_mun["n_vut"] + df_mun["n_habitaciones"]
    df_mun["total_plazas_reate"] = df_mun["plazas_vut"] + df_mun["plazas_habitaciones"]
    df_mun["plazas_reate_por_1000_hab"] = (df_mun["total_plazas_reate"] / df_mun["pob_2025"]) * 1000
    df_mun["plazas_vut_por_1000_hab"] = (df_mun["plazas_vut"] / df_mun["pob_2025"]) * 1000
    df_mun["plazas_trad_por_1000_hab"] = (df_mun["plazas_aloj_tradicionales"] / df_mun["pob_2025"]) * 1000
    df_mun["pct_plazas_habitacion"] = np.where(
        df_mun["total_plazas_reate"] > 0,
        (df_mun["plazas_habitaciones"] / df_mun["total_plazas_reate"]) * 100,
        0.0,
    )
    df_mun["crec_pob_2016_2025_pct"] = ((df_mun["pob_2025"] - df_mun["pob_2016"]) / df_mun["pob_2016"]) * 100
    df_mun["peso_renta_inmob_pct"] = (df_mun["renta_cap_inmobiliario_2024"] / df_mun["renta_total_2024"]) * 100

    mun_ctx = df_mun[["cod_mun", "pob_2025", "renta_total_2024", "renta_disponible_2024"]]
    df_reate_enriched = df_reate.merge(mun_ctx, on="cod_mun", how="left")

    df_mun.to_csv(out_dir / "dataset_municipal.csv", index=False, encoding="utf-8-sig")
    df_reate_enriched.to_csv(out_dir / "reate_unificado.csv", index=False, encoding="utf-8-sig")
    return df_mun, df_reate_enriched


if __name__ == "__main__":
    df_mun, df_reate = build_datasets(DATA_DIR, OUT_DIR)
    print("dataset_municipal.csv generado:", df_mun.shape)
    print("reate_unificado.csv generado:", df_reate.shape)