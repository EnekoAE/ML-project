from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "libreria"
OUT_DIR = Path(__file__).resolve().parent


def normalize_key(text: str) -> str:
    if pd.isna(text):
        return ""
    s = str(text)
    try:
        s = s.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    s = s.lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[/\-\(\)\.,]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_raw_dataset(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    df_listings = pd.read_csv(data_dir / "listings.csv", low_memory=False)
    df_mun = pd.read_csv(out_dir / "dataset_municipal.csv", dtype={"cod_mun": str})
    df_reate = pd.read_csv(out_dir / "reate_unificado.csv", dtype={"cod_mun": str})

    lookup = {}
    for _, row in df_mun.iterrows():
        code = str(row["cod_mun"]).zfill(5)
        mun = str(row["municipio"])
        lookup[normalize_key(mun)] = code
        if "/" in mun:
            for part in mun.split("/"):
                p_norm = normalize_key(part)
                if p_norm and p_norm not in lookup:
                    lookup[p_norm] = code

    aliases = {
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
        "donostia": "20069",
        "san sebastian": "20069",
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
    lookup.update(aliases)

    mun_col = "neighbourhood_cleansed" if "neighbourhood_cleansed" in df_listings.columns else "neighbourhood"
    df_listings["cod_mun"] = df_listings[mun_col].apply(normalize_key).map(lookup)

    cols_eustat = [
        "cod_mun",
        "pob_2025",
        "crec_pob_2016_2025_pct",
        "renta_total_2024",
        "renta_disponible_2024",
        "peso_renta_inmob_pct",
        "n_vut",
        "n_habitaciones",
        "plazas_reate_por_1000_hab",
        "n_aloj_tradicionales",
        "plazas_trad_por_1000_hab",
    ]
    cols_eustat_ok = [c for c in cols_eustat if c in df_mun.columns]
    df_raw = df_listings.merge(df_mun[cols_eustat_ok], on="cod_mun", how="left")

    reate_sub = df_reate[["n_registro", "categoria_reate", "superficie_m2", "nucleo_urbano", "anio_inscripcion"]].copy()
    reate_sub["n_registro_key"] = reate_sub["n_registro"].astype(str).str.strip().str.upper()
    reate_sub = reate_sub.drop_duplicates(subset=["n_registro_key"])

    df_raw["licencia_key"] = (
        df_raw["license"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.extract(r"\b([EL](?:BI|SS|VI)\d{4,6})\b", expand=False)
        .fillna("")
    )
    df_raw = df_raw.merge(
        reate_sub.drop(columns=["n_registro"]),
        left_on="licencia_key",
        right_on="n_registro_key",
        how="left",
    ).drop(columns=["licencia_key", "n_registro_key"])

    df_raw.to_csv(out_dir / "dataset_euskadi_raw.csv", index=False, encoding="utf-8-sig")
    return df_raw


if __name__ == "__main__":
    df_raw = build_raw_dataset(DATA_DIR, OUT_DIR)
    print("dataset_euskadi_raw.csv generado:", df_raw.shape)