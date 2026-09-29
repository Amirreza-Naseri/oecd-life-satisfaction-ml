# Publishing this project on GitHub

## 1. Create a new repository

Recommended repository name:

```text
oecd-life-satisfaction-ml
```

Use this short description:

```text
Leakage-safe machine learning pipeline for predicting OECD life satisfaction with cross-validation, feature selection, baseline models, and an MLP regressor.
```

Set the repository to **Public** if you want recruiters to see it. Do not initialize it with a README, `.gitignore`, or license because those files already exist here.

## 2. Upload with Git

Open a terminal inside this project folder:

```bash
git init
git add .
git commit -m "Initial portfolio-ready OECD life satisfaction project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/oecd-life-satisfaction-ml.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## 3. Repository settings

Add these topics on GitHub:

```text
machine-learning
data-science
python
scikit-learn
neural-network
regression
oecd
cross-validation
feature-selection
```

Pin this repository on your GitHub profile if Data Science / Machine Learning roles are your target.

## 4. Before sharing with recruiters

Check that the README renders the result plot correctly, the Actions tab shows passing tests, and the repository does not contain `.venv`, cache folders, or local absolute paths.

## CV bullet

A concise CV version:

> **OECD Life Satisfaction Prediction** — Built a reproducible regression workflow on OECD Better Life Index data using leakage-safe preprocessing, feature selection, 10-fold cross-validation, baseline models, and an MLP neural network; the recorded original experiment achieved RMSE 0.607 and MAE 0.445 across 41 countries.

## Interview talking points

Be ready to explain why preprocessing belongs inside cross-validation, why a neural network may not outperform simpler models on only ~41 country-level observations, why RMSE/MAE are preferable to calling ±0.5 tolerance “accuracy,” and how you would extend the project using nested CV or explainability.
