# Loan-Predication
Loan Credibility prediction on SBA data

---

## 🧠 Problem Statement:

We are working with a loan dataset (SBA National data) and our goal is to:
> ✅ Predict whether a loan will default or not,  
while also considering the business cost impact of wrong predictions.

This is a classification problem, but instead of just using typical accuracy/F1 scores, we also wanted to evaluate models based on:
- Net Profit (penalize false positives more heavily),
- F1 Score (balance between precision & recall),
- AUC Score (overall classification performance).

---

## 🛠️ Steps in the Code:

### 🔹 1. Data Loading & Preprocessing
- Load the dataset (e.g., SBA loans).
- Clean columns like `'DisbursementGross'`, `'ChgOffPrinGr'` by removing `$`, `,` and converting to float.
- Create a new binary target column:
  ```python
  df['Default'] = df['ChgOffPrinGr'].astype(float) > 0
  ```

---

### 🔹 2. Feature Engineering
- Apply one-hot encoding using:
  ```python
  pd.get_dummies(df, drop_first=True)
  ```
- Split data into:
  - `X` → features
  - `y` → target (`Default`)
  - `amount` → original loan amount (`DisbursementGross`) for profit calculation

---

### 🔹 3. Train/Test Split
- Use `train_test_split()` to divide the data into training and validation sets.

---

### 🔹 4. Cost-Sensitive Evaluation
Define a custom cost-sensitive metric:  
```python
Net Profit = True Positive  0 + True Negative  0 
             - False Positive  Loan Amount 
             - False Negative  0
```
So we mainly penalize:
- False Positives → wrongly approved risky loans.

Defined in:
```python
def compute_net_profit(y_true, y_pred, amount):
    # Subtract amounts where default was predicted incorrectly
```

---

### 🔹 5. Model Evaluation Function
Create a generic function to:
- Train any model,
- Get predicted probabilities (or decision scores),
- Classify based on a threshold (0.5),
- Calculate:
  - F1 Score
  - AUC
  - Net Profit

```python
def run_model(model, model_name):
    ...
    return model_name, net_profit, f1, auc
```

It includes logic to handle models that don’t support `.predict_proba()` like `RidgeClassifier`.

---

### 🔹 6. Model Training
Train and evaluate multiple classifiers:
```python
models = [
    (LogisticRegression(), "Logistic Regression"),
    (RandomForestClassifier(), "Random Forest"),
    ...
]
for model, name in models:
    results.append(run_model(model, name))
```

---

### 🔹 7. Final Comparison Table
Display all models ranked by Net Profit:
```python
result_df = pd.DataFrame(results, columns=['Model', 'Net Profit', 'F1 Score', 'AUC'])
result_df.sort_values(by='Net Profit', ascending=False, inplace=True)
print(result_df)
```

---

## ✅ Final Goal Achieved:
By the end, you had:
- Multiple models evaluated,
- Each scored not just on accuracy metrics,
- But also on financial business impact (Net Profit),
- With results in a clean comparison table.

---

