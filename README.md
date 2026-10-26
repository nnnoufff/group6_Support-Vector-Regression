# Support Vector Regression (SVR): Predicting Medical Insurance Expenses

**Group 6 · Regression Algorithms Exploration (mini-project)**

Can Support Vector Regression predict a person's medical expenses from their age, sex, BMI, number of children, smoking status and region? This project explores SVR end to end: exploratory data analysis, preprocessing, model training, tuning and evaluation.

## Dataset

- **Source:** [Insurance Premium Prediction (Kaggle)](https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction)
- **Size:** 1,337 customers, 6 features, 1 target, no missing values
- **Target:** `expenses` (continuous, USD)

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Age of the customer |
| `sex` | categorical | Male / female |
| `bmi` | numeric | Body mass index |
| `children` | numeric | Number of children covered |
| `smoker` | categorical | Smoker / non-smoker |
| `region` | categorical | Residential region in the US |
| `expenses` | numeric | Medical expenses charged (target) |

## Project Workflow

1. **Exploratory Data Analysis (EDA)**: distributions, outliers, relationships between features and the target.
2. **Preprocessing**: encoding categorical features, scaling, train/test split.
3. **Modelling**: SVR (RBF kernel) compared with a Linear Regression baseline.
4. **Tuning and evaluation**: grid search over `C`, `epsilon` and `gamma`; MAE and R².

## Key EDA Findings

- `expenses` is right-skewed (mean ≈ 13.3k vs median ≈ 9.4k).
- 139 outliers, about 98% of them smokers. They are valid cases, so they were **kept**.
- **Smoking is the strongest factor:** smokers pay about 3.8× more (32.1k vs 8.4k on average).
- **Age** raises expenses steadily for everyone (9.2k → 17.9k from ages 18–29 to 50–64).
- **BMI** matters mainly for smokers (21.4k below BMI 30 vs 41.6k at BMI 30 or more). This motivated the engineered `obese` feature (`BMI ≥ 30`).
- Children, sex and region show little or no clear effect.

## Results

| Model | MAE | R² |
|---|---|---|
| SVR, default | n/a | −0.131 |
| SVR, scaled target | 2,180 | 0.893 |
| **SVR, tuned** (`C=100`, `epsilon=0.1`, `gamma=0.01`) | **2,098** | **0.909** |
| Linear Regression (baseline) | 4,288 | 0.806 |

## Repository Structure

```
├── README.md
├── MiniProject_Group6.ipynb   # main notebook (EDA, modelling, evaluation)
└── presentation/              # slides
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook MiniProject_Group6.ipynb
```

Download `insurance.csv` from the Kaggle link above and place it where the notebook expects it.

## Team (Group 6):
- Nouf Turki
- Nouf Alotaibi
- Mohammed
- Razan
