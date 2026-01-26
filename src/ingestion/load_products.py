from pathlib import Path
from typing import Iterable, Union

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "products.csv"


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_products(path: Union[str, Path] = DATA_PATH) -> pd.DataFrame:
    """Load, validate, and normalize product catalog with financial metadata."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Products file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {
        "id",
        "title",
        "description",
        "price",
        "category",
        "brand",
        "rating",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=list(required_columns))

    numeric_columns = [
        "id",
        "price",
        "rating",
        "msrp",
        "discount_pct",
        "stock",
        "max_installments",
        "shipping_days",
    ]
    df = _coerce_numeric(df, numeric_columns)

    df = df[df["price"] > 0]
    df = df[(df["rating"] >= 0) & (df["rating"] <= 5)]
    if "discount_pct" in df.columns:
        df = df[(df["discount_pct"].isna()) | ((df["discount_pct"] >= 0) & (df["discount_pct"] <= 100))]
    if "stock" in df.columns:
        df = df[df["stock"].isna() | (df["stock"] >= 0)]

    df["title"] = df["title"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip().str.title()
    df["brand"] = df["brand"].astype(str).str.strip()

    if "payment_methods" in df.columns:
        df["payment_methods"] = df["payment_methods"].fillna("").apply(
            lambda x: [m.strip().lower() for m in str(x).split(";") if m.strip()]
        )

    if "installment_available" in df.columns:
        df["installment_available"] = (
            df["installment_available"].astype(str).str.lower().isin(["true", "1", "yes"])
        )

    if "budget_band" in df.columns:
        allowed_bands = {"budget", "midrange", "premium"}
        df["budget_band"] = df["budget_band"].astype(str).str.lower().where(
            df["budget_band"].astype(str).str.lower().isin(allowed_bands), other=pd.NA
        )

    if "tags" in df.columns:
        df["tags"] = df["tags"].fillna("").apply(
            lambda x: [t.strip().lower() for t in str(x).split(";") if t.strip()]
        )

    if not df["id"].is_unique:
        raise ValueError("Product IDs must be unique")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    products = load_products()
    print(f"Loaded {len(products)} products successfully")
    print(products.head())
