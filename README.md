# 🩺 Medical Equipment Cost Prediction

This repository contains our submission for the **[Medical Equipments Cost Prediction Challenge](https://www.kaggle.com/competitions/Medical-Equipments-Cost-Prediction-Challenge)** on Kaggle.  
The task is to predict the **transportation cost** of medical equipment based on supplier, equipment, and shipment details using machine learning models.

---

## 📘 Overview

The project aims to build a robust **regression pipeline** that accurately estimates medical transport costs.  
We explored various preprocessing strategies, handled data irregularities like missing values and outliers, and compared several models — ranging from simple linear regression to ensemble methods — to identify the best-performing approach.

---

## 🧩 Key Steps

### 🔹 1. Data Cleaning

- Removed identifier columns with no predictive value (`Hospital_Id`, `Supplier_Name`, etc.).  
- Handled missing data using mode, mean, and grouped imputations.  
- Outliers trimmed using percentile thresholds (e.g., 5th–95th).

### 🔹 2. Feature Engineering

- Extracted new features like **transport duration** from datetime columns.  
- Handled categorical features via **One-Hot Encoding**.  
- Scaled numerical features for stable optimization.  
- Dropped highly correlated columns to reduce redundancy.

### 🔹 3. Model Building

We tested multiple regression techniques:

- **Linear, Lasso, Ridge, ElasticNet, Bayesian Ridge**
- **Polynomial Regression** (degree 2–4)
- **Tree-based models:** Random Forest, XGBoost, Gradient Boosting, AdaBoost  

After hyperparameter tuning, the **Polynomial Ridge Regression (degree = 3)** model achieved the best performance.

---

## 🧮 Experiments

Over 100+ experiments were conducted with different preprocessing and model combinations.  
Best results were obtained with:

- Polynomial features (degree 3)  
- Ridge regression regularization  
- Outlier filtering between 10th–95th percentile  
- Group-based missing value imputation  

**Final Kaggle Public Score:** `3.47e9 (MSE)`

---

## 📈 Insights

- Most numerical columns (Height, Width, Weight, Value) are **right-skewed** with **high-value outliers**.  
- Tree-based models underperformed due to sparse categorical features and small dataset size.  
- Polynomial Ridge regression generalized better, capturing non-linear feature relationships.

---

## ⚙️ How to Run

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your-username>/medical-equipment-cost-prediction.git
   cd medical-equipment-cost-prediction


2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**

   ```bash
   jupyter notebook ML_Assignment_1.ipynb
   ```

   Execute all cells to reproduce results or modify preprocessing/model parameters.

---

## 🧠 Tech Stack

- **Language:** Python 3.x
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`
- **Environment:** Jupyter Notebook

---

## 🏁 Results Summary

| Model                    | Key Technique                             | Score (MSE) |
| ------------------------ | ----------------------------------------- | ----------- |
| Polynomial Ridge (deg=3) | Group mean imputation + outlier filtering | **3.47e9**  |
| Polynomial Ridge (deg=4) | Slightly overfit                          | 3.65e9      |
| Bayesian Ridge           | Stable but less accurate                  | 5.3e9       |
| Random Forest            | High variance                             | 6.1e9       |

---

## 👥 Authors

- **[Navaneeth D (IMT2023095)](https://github.com/ar3s-nd)**
- **[Pramatha V Rao (IMT2023116)](github.com/p-r-a-o)**
- **[Bharat Kumar (IMT2023504)](https://github.com/BharatDhulgond)**

📅 *Submitted on:* 25 October 2025
📘 *Course:* AIT-511 (Machine Learning) – IIIT Bangalore
