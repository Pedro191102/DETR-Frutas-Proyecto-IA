from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import slugify

NUTRIENT_COLUMNS = [
    "kcal_100g", "protein_g_100g", "fat_g_100g", "carbs_g_100g",
    "fiber_g_100g", "water_g_100g",
]


def load_reference(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"product", "weight_avg_g", *NUTRIENT_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Nutrition file is missing columns: {sorted(missing)}")
    if "aliases" not in df.columns:
        df["aliases"] = ""
    df["key"] = df["product"].astype(str).map(slugify)
    for col in ["weight_avg_g", *NUTRIENT_COLUMNS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    alias_to_key: dict[str, str] = {}
    for _, row in df.iterrows():
        alias_to_key[row["key"]] = row["key"]
        for alias in str(row.get("aliases", "")).split(";"):
            alias_key = slugify(alias)
            if alias_key:
                alias_to_key[alias_key] = row["key"]
    df = df.set_index("key", drop=False)
    df.attrs["alias_to_key"] = alias_to_key
    return df


def estimate_selected_rows(rows: list[dict[str, Any]], reference: pd.DataFrame) -> dict[str, Any]:
    """Compute rough totals using references and user-confirmed portions.

    No weight is inferred from pixels. Incomplete portions require either a
    manual weight or an explicit fraction entered by the user.
    """
    totals = {col: 0.0 for col in NUTRIENT_COLUMNS}
    nutrient_complete = {col: True for col in NUTRIENT_COLUMNS}
    total_weight = 0.0
    weight_complete = True
    contributions = []
    aliases = reference.attrs.get("alias_to_key", {})
    for row in rows:
        if not bool(row.get("include", True)):
            continue
        label = str(row.get("label", ""))
        key = aliases.get(slugify(label), slugify(label))
        if key not in reference.index:
            weight_complete = False
            for col in NUTRIENT_COLUMNS:
                nutrient_complete[col] = False
            contributions.append({"label": label, "weight_g": None, "note": "Producto sin fila de referencia"})
            continue
        ref = reference.loc[key]
        manual = row.get("manual_weight_g")
        fraction = row.get("portion_fraction")
        state = str(row.get("portion_status", "Completa"))
        weight = None
        if manual is not None and pd.notna(manual) and float(manual) > 0:
            weight = float(manual)
        elif state == "Completa" and pd.notna(ref["weight_avg_g"]):
            weight = float(ref["weight_avg_g"])
        elif state in {"Parcial", "Cortada"} and fraction is not None and pd.notna(fraction) and pd.notna(ref["weight_avg_g"]):
            fraction_value = float(fraction)
            if 0 < fraction_value <= 1:
                weight = float(ref["weight_avg_g"]) * fraction_value
        if weight is None:
            weight_complete = False
            for col in NUTRIENT_COLUMNS:
                nutrient_complete[col] = False
            contributions.append({"label": label, "weight_g": None, "note": "Falta peso medio, fracción o peso manual"})
            continue
        total_weight += weight
        missing_nutrients = []
        for col in NUTRIENT_COLUMNS:
            value = ref[col]
            if pd.notna(value):
                totals[col] += float(value) * weight / 100.0
            else:
                nutrient_complete[col] = False
                missing_nutrients.append(col)
        contributions.append({
            "label": label,
            "weight_g": weight,
            "note": "Estimación por referencia" if not missing_nutrients else "Peso estimado; faltan algunos nutrientes",
        })
    return {
        "total_weight_g": total_weight if weight_complete else None,
        "known_weight_subtotal_g": total_weight,
        "weight_complete": weight_complete,
        "nutrients": {col: totals[col] if nutrient_complete[col] else None for col in NUTRIENT_COLUMNS},
        "nutrient_complete": nutrient_complete,
        "contributions": contributions,
    }
