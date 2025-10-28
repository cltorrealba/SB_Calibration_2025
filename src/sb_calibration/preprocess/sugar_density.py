"""Simple sugar-density pipeline (port of legacy `sugar_density.py` core).

This module exposes a small API: `fit_density_model(df, x_col, y_col)` that fits a
Polynomial + LinearRegression pipeline and returns the fitted pipeline and coeffs.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression


def fit_density_model(df: pd.DataFrame, x_col: str = "sugar_pct", y_col: str = "density_g_mL", degree: int = 3) -> Tuple[Pipeline, np.ndarray]:
    X = df[[x_col]].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lin", LinearRegression()),
    ])
    pipeline.fit(X, y)
    # Extract coefficients from linear regression (after poly)
    lin = pipeline.named_steps["lin"]
    coefs = np.concatenate(([lin.intercept_], lin.coef_.ravel()))
    return pipeline, coefs


def load_dataset(path: str, x_col: str = "sugar_pct", y_col: str = "density_g_mL") -> pd.DataFrame:
    df = pd.read_csv(path)
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Expected columns {x_col} and {y_col} in dataset")
    return df


def save_coeffs(coefs: np.ndarray, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for c in coefs:
            fh.write(f"{float(c)}\n")


def run_from_csv(input_csv: str, out_coeffs: str, x_col: str = "sugar_pct", y_col: str = "density_g_mL", degree: int = 3):
    df = load_dataset(input_csv, x_col=x_col, y_col=y_col)
    pipeline, coefs = fit_density_model(df, x_col=x_col, y_col=y_col, degree=degree)
    save_coeffs(coefs, out_coeffs)
    return pipeline, coefs
