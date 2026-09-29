# Results

Files prefixed with `original_notebook_` were reconstructed directly from the completed output stored in the user's original notebook. They are included as a traceable benchmark.

A fresh run of the refactored pipeline creates `model_comparison.csv`, `cv_predictions.csv`, `selected_features.csv`, and three PNG diagnostics in this folder. Those fresh metrics may differ slightly because imputation is now performed inside each cross-validation fold.
