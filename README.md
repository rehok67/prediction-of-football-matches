Football Surprise Outcome Prediction with Regression

This project aims to identify **"surprise" outcomes in football matches** — instances where the result defied market expectations — using logistic regression. It explores inefficiencies in betting markets by analyzing historical odds and match data across 15+ years of league football.

Project Overview

Rather than classifying football outcomes into the standard three-way format (Home Win, Draw, Away Win), this model reframes the task into a **binary classification problem**:

- `1` → The outcome was a **surprise** (the least likely result occurred)
- `0` → The **market prediction was correct** (expected result)

The focus is to **learn patterns** where betting odds failed to reflect the actual outcome, helping uncover value opportunities in betting strategies.

## 📊 Features Used

The model is trained on a rich set of pre-match data, including:

- 🧮 **Bookmaker odds** (excluding the highest-odds outcome)
- 📉 **Expected Goals (xG)** predictions before the match
- 📊 **Statistical prediction models** and consensus tips
- 📆 **Match metadata** (league, date, teams)
- 🏁 **Actual match result**

> Note: The "market prediction" is based on the lowest and mid-range odds (typically associated with the favored team and the draw), while the highest odds (typically the underdog) are interpreted as potential surprise signals.

## 🗃️ Dataset

- Historical match data from top football leagues over the past **15+ seasons**
- Cleaned and preprocessed to include relevant numeric and categorical features
- Integrated multiple data sources (odds markets, prediction models, match stats)

## 🎯 Goal

The goal is to **train a machine learning model** that can generalize and identify patterns where **bookmaker predictions failed** — potentially aiding bettors, analysts, or researchers in detecting value bets.

## 🛠️ Tech Stack

- **Python**
- **scikit-learn** for logistic regression
- **Pandas, NumPy** for data manipulation
- **Matplotlib, Seaborn** for exploratory data analysis
- **Jupyter Notebooks** for experiments

## 📈 Model Performance

The model is evaluated using metrics like:

- Accuracy
- Precision/Recall for the "surprise" class
- ROC-AUC to capture imbalance in class distribution

Balanced class representation and feature scaling techniques were used to ensure the model doesn’t favor the dominant class.


## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is **not intended as financial advice or betting guidance**. Betting involves risk and should be approached responsibly.



---
