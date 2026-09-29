import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import build_models, regression_metrics


def test_all_models_keep_imputation_inside_pipeline():
    models = build_models(n_features=6, k_features=4)
    for model in models.values():
        assert list(model.named_steps)[:3] == ["imputer", "scaler", "kbest"]


def test_regression_metrics_perfect_prediction():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(y, np.array([1.0, 2.0, 3.0, 4.0]))
    assert metrics["RMSE"] == 0.0
    assert metrics["MAE"] == 0.0
    assert metrics["R2"] == 1.0
    assert metrics["Within_±0.5"] == 100.0



def test_loader_accepts_legacy_value_schema(tmp_path):
    from src.train import load_and_prepare_data

    rows = []
    for country, target, x1 in [("A", 6.0, 1.0), ("B", 7.0, 2.0), ("OECD - Total", 6.5, 1.5)]:
        rows.extend([
            {"Country": country, "INDICATOR": "SW_LIFS", "Indicator": "Life satisfaction", "INEQUALITY": "TOT", "Inequality": "Total", "Value": target},
            {"Country": country, "INDICATOR": "X1", "Indicator": "Feature 1", "INEQUALITY": "TOT", "Inequality": "Total", "Value": x1},
        ])
    path = tmp_path / "legacy.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    X, y, mapping = load_and_prepare_data(path)
    assert list(X.index) == ["A", "B"]
    assert X.shape == (2, 1)
    assert y.tolist() == [6.0, 7.0]
    assert mapping["X1"] == "Feature 1"
