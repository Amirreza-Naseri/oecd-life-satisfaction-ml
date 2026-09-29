from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

TARGET_COL = "SW_LIFS"
RANDOM_STATE = 42


def load_and_prepare_data(
    csv_path: str | Path,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """Load an OECD DF_BLI export and reshape it to country x indicator format.

    The historical OECD export appears in two common schemas: older downloads use
    ``Value`` while newer Data Explorer exports may use ``OBS_VALUE``. This loader
    accepts both. It also accepts either the label ``Inequality == 'Total'`` or
    the code ``INEQUALITY == 'TOT'``.

    Missing feature values are intentionally left untouched here; imputation is
    fitted inside each cross-validation fold to avoid leakage.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Run `python src/download_data.py` first."
        )

    df = pd.read_csv(csv_path)
    required = {"Country", "INDICATOR", "Indicator"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    value_col = "OBS_VALUE" if "OBS_VALUE" in df.columns else "Value" if "Value" in df.columns else None
    if value_col is None:
        raise ValueError("Dataset must contain either 'OBS_VALUE' or 'Value'.")

    if "Inequality" in df.columns:
        total_mask = df["Inequality"].astype(str).str.strip().eq("Total")
    elif "INEQUALITY" in df.columns:
        total_mask = df["INEQUALITY"].astype(str).str.strip().eq("TOT")
    else:
        raise ValueError("Dataset must contain either 'Inequality' or 'INEQUALITY'.")

    df_tot = df.loc[total_mask].copy()
    if df_tot.empty:
        raise ValueError("No Total/TOT rows were found in the dataset.")

    df_tot[value_col] = pd.to_numeric(df_tot[value_col], errors="coerce")
    pivot = df_tot.pivot_table(
        index="Country",
        columns="INDICATOR",
        values=value_col,
        aggfunc="mean",
    )

    if target_col not in pivot.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found after pivoting. "
            f"Available indicators include: {list(pivot.columns[:10])}"
        )

    pivot = pivot.dropna(subset=[target_col])
    pivot = pivot.loc[pivot.index != "OECD - Total"]

    X = pivot.drop(columns=[target_col]).astype(float)
    y = pivot[target_col].astype(float)

    code_to_name = (
        df_tot.drop_duplicates("INDICATOR")
        .set_index("INDICATOR")["Indicator"]
        .astype(str)
        .to_dict()
    )
    return X, y, code_to_name


def _shared_preprocessing(k: int) -> list[tuple[str, object]]:
    return [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler()),
        ("kbest", SelectKBest(score_func=f_regression, k=k)),
    ]


def build_models(n_features: int, k_features: int = 10) -> Dict[str, Pipeline]:
    """Create leakage-safe regression pipelines used in the experiment."""
    if n_features < 1:
        raise ValueError("At least one feature is required.")

    k = min(k_features, n_features)

    return {
        "Ridge": Pipeline(
            _shared_preprocessing(k)
            + [("model", Ridge(alpha=1.0))]
        ),
        "Random Forest": Pipeline(
            _shared_preprocessing(k)
            + [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "Neural Network": Pipeline(
            _shared_preprocessing(k)
            + [
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(16, 8),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        learning_rate_init=0.01,
                        max_iter=10_000,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=50,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    tolerance: float = 0.5,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    if np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1])

    within_tolerance = float(
        100.0 * np.mean(np.abs(y_pred_arr - y_true_arr) <= tolerance)
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson_r": pearson_r,
        "Pearson_r_squared": pearson_r**2 if not np.isnan(pearson_r) else float("nan"),
        f"Within_±{tolerance}": within_tolerance,
    }


def evaluate_models(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 10,
    tolerance: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Generate out-of-fold predictions and comparable regression metrics."""
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if len(y) < 3:
        raise ValueError("At least 3 samples are required for cross-validation.")

    splits = min(n_splits, len(y))
    if splits < 2:
        raise ValueError("Cross-validation requires at least 2 folds.")

    cv = KFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}

    for name, model in models.items():
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
        predictions[name] = pred
        row: dict[str, float | str] = {"Model": name}
        row.update(regression_metrics(y, pred, tolerance=tolerance))
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    return results, predictions


def selected_features(
    neural_network: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    code_to_name: Dict[str, str],
) -> pd.DataFrame:
    fitted = clone(neural_network).fit(X, y)
    support = fitted.named_steps["kbest"].get_support()
    scores = fitted.named_steps["kbest"].scores_

    rows = []
    for code, keep, score in zip(X.columns, support, scores):
        if keep:
            rows.append(
                {
                    "IndicatorCode": code,
                    "Indicator": code_to_name.get(code, ""),
                    "FScore": float(score) if score is not None else float("nan"),
                }
            )

    return pd.DataFrame(rows).sort_values("FScore", ascending=False).reset_index(drop=True)


def save_artifacts(
    output_dir: str | Path,
    countries: pd.Index,
    y: pd.Series,
    results: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    features: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "model_comparison.csv", index=False)
    features.to_csv(output_dir / "selected_features.csv", index=False)

    pred_df = pd.DataFrame({"Country": countries, "Actual": y.to_numpy()})
    for model_name, pred in predictions.items():
        pred_df[model_name] = pred
    pred_df.to_csv(output_dir / "cv_predictions.csv", index=False)

    # Plot 1: model comparison (RMSE; lower is better)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = results.sort_values("RMSE", ascending=False)
    ax.barh(ordered["Model"], ordered["RMSE"])
    ax.set_xlabel("Cross-validated RMSE")
    ax.set_title("Model Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    # Plot 2: actual vs predicted for the neural network
    nn_pred = predictions["Neural Network"]
    lo = float(min(y.min(), np.min(nn_pred)))
    hi = float(max(y.max(), np.max(nn_pred)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, nn_pred, alpha=0.8)
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Actual life satisfaction")
    ax.set_ylabel("Out-of-fold prediction")
    ax.set_title("Neural Network: Actual vs Predicted")
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    # Plot 3: residuals for the neural network
    residuals = y.to_numpy() - nn_pred
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(nn_pred, residuals, alpha=0.8)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Out-of-fold prediction")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Neural Network Residuals")
    fig.tight_layout()
    fig.savefig(output_dir / "residuals.png", dpi=180)
    plt.close(fig)


def run_experiment(
    data_path: str | Path,
    output_dir: str | Path = "results",
    target_col: str = TARGET_COL,
    k_features: int = 10,
    n_splits: int = 10,
    tolerance: float = 0.5,
) -> pd.DataFrame:
    X, y, code_to_name = load_and_prepare_data(data_path, target_col=target_col)
    models = build_models(n_features=X.shape[1], k_features=k_features)

    results, predictions = evaluate_models(
        models,
        X,
        y,
        n_splits=n_splits,
        tolerance=tolerance,
    )

    features = selected_features(models["Neural Network"], X, y, code_to_name)
    save_artifacts(output_dir, X.index, y, results, predictions, features)

    print(f"Countries: {len(X)}")
    print(f"Candidate indicators: {X.shape[1]}")
    print(f"Target: {target_col}")
    print("\nCross-validated model comparison:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nArtifacts written to: {Path(output_dir).resolve()}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OECD Better Life Index life-satisfaction regression experiment."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/oecd_bli.csv"),
        help="Path to the OECD Better Life Index CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for metrics, predictions, and plots.",
    )
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--k-features", type=int, default=10)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        data_path=args.data,
        output_dir=args.output_dir,
        target_col=args.target,
        k_features=args.k_features,
        n_splits=args.folds,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
