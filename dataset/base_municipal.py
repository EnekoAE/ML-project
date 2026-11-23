from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "libreria"
OUT_DIR = Path(__file__).resolve().parent


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[/\-\(\)\.,]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_population(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", skiprows=3, encoding="latin1")
    start_idx = raw[raw["Entidad"] == "Detalle por municipios"].index[0]
    df = raw.iloc[start_idx + 2 :].copy()
    df = df.rename(columns={"Entidad": "cod_mun", "Unnamed: 1": "municipio"})
    df["cod_mun"] = df["cod_mun"].astype(str).str.strip().str.zfill(5)
    df["municipio"] = df["municipio"].astype(str).str.strip()

    prov_map = {"01": "Araba/Álava", "20": "Gipuzkoa", "48": "Bizkaia"}
    df["cod_prov"] = df["cod_mun"].str[:2]
    df["provincia"] = df["cod_prov"].map(prov_map)

    year_cols = [str(y) for y in range(2001, 2026)]
    for col in year_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    df = df.rename(columns={y: f"pob_{y}" for y in year_cols})
    keep_cols = ["cod_mun", "cod_prov", "provincia", "municipio"] + [f"pob_{y}" for y in year_cols]
    return df[keep_cols].reset_index(drop=True)


def load_income(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    mun_start = [i for i, l in enumerate(lines) if l.strip() == '"Municipios"'][0]
    mun_end = [i for i, l in enumerate(lines) if l.strip().startswith('"Fuente:')][0]

    rows = [
        [p.strip('"') for p in l.strip().split(";")]
        for l in lines[mun_start + 1 : mun_end]
        if l.strip()
    ]
    cols = [
        "municipio_inc",
        "renta_total_2024",
        "renta_trabajo_2024",
        "renta_cap_mobiliario_2024",
        "renta_cap_inmobiliario_2024",
        "renta_actividades_2024",
        "transferencias_2024",
        "renta_disponible_2024",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for col in cols[1:]:
        df[col] = pd.to_numeric(
            df[col].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    alias_map = {
        "arrankudiaga zollo": "arrankudiaga",
        "ribera baja erriberabeitia": "ribera baja erribera beitia",
        "agurain salvatierra": "salvatierra agurain",
    }
    df["norm_name"] = df["municipio_inc"].apply(normalize_name).replace(alias_map)
    return df


def build_base_municipal(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    df_pop = load_population(data_dir / "indicator.csv")
    df_inc = load_income(data_dir / "xls0005791_c.csv")

    df_pop["norm_name"] = df_pop["municipio"].apply(normalize_name)
    df_base = df_pop.merge(df_inc.drop(columns=["municipio_inc"]), on="norm_name", how="left")
    df_base = df_base.drop(columns=["norm_name"])

    assert len(df_base) == 252
    assert df_base["cod_mun"].nunique() == 252
    assert df_base[["pob_2025", "renta_total_2024", "renta_disponible_2024"]].isna().sum().sum() == 0

    df_base.to_csv(out_dir / "base_municipal.csv", index=False, encoding="utf-8-sig")
    return df_base


if __name__ == "__main__":
    df_base = build_base_municipal(DATA_DIR, OUT_DIR)
    print("base_municipal.csv generado correctamente:", df_base.shape)