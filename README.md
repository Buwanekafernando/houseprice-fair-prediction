# 🏠 House Price Fairness Prediction

This project focuses on **data preprocessing, exploratory data analysis (EDA), and machine learning modeling** to evaluate the **fairness of house prices** in Sri Lanka and India.  
The pipeline handles raw housing datasets, cleans and encodes features, and applies regression models to classify houses as **Fair, Overpriced, or Underpriced**.

---

## 📌 Project Overview
The project is divided into four main phases:
1. **Preprocessing** – Data cleaning, handling missing values, unit conversions, and feature engineering.
2. **Outlier Removal** – Applying IQR-based filtering and visualizing before/after distributions.
3. **Exploratory Data Analysis (EDA)** – Distributions, scatter plots, and categorical breakdowns of features.
4. **Model Training & Fairness Prediction** – Training a **Random Forest Regressor** and using rules/ML predictions to classify property prices.

---

## 🚀 Features
- Cleans and standardizes raw real-estate datasets.
- Converts inconsistent price formats (Lakh/Cr) → **Rupees**.
- Extracts numeric values from messy text fields (e.g., Carpet Area, Parking).
- Encodes **categorical values** such as Furnishing, Parking Type, and Location.
- Removes **outliers** using the IQR method.
- Visualizes distributions and correlations via **Matplotlib & Seaborn**.
- Trains ML models for predicting house prices.
- Provides **fairness labels** (`Fair`, `Overpriced`, `Underpriced`) based on baseline rules and regression outputs.

---

## 📂 Dataset
- **Input:** `house_prices.csv`  
- **Cleaned output:** `house_prices_cleaned.csv` → `cleaned_data.csv`  

Columns include:
- `Location`, `Super_Area`, `Carpet_Area`, `Bathroom`, `Balcony`, `Furnishing`, `Car_Parking`, `Ownership`, `Facing`, `Overlooking`, `Status`, `Transaction`, `Amount_in_rupees`.

---

## ⚙️ Workflow
### 1. Preprocessing
- Dropped duplicates and irrelevant columns.
- Converted **Price** from “Lac/Cr” to rupees.
- Extracted numeric values from Carpet/Super area.
- Encoded **Car Parking** as: `0=None, 1=Open, 2=Covered`.
- Grouped **locations** into:
  - Super Urban = 4
  - Super Urban Suburb = 3
  - Urban = 2
  - Rural/Small Town = 1
  - Other/Unknown = 0

### 2. Outlier Removal
- Applied IQR filtering on numeric features.
- Boxplots before & after cleaning.

### 3. Exploratory Data Analysis
- Distribution plots for price, area, parking, etc.
- Scatter plots: Price vs Area.
- KDE plots for density visualization.
- Categorical count plots.

### 4. Model Training
- Features: `Location, Super_Area, Carpet_Area, Bathroom, Furnishing, Car_Parking, Balcony, Overlooking`.
- Target: `Amount_in_rupees`.
- Model: **RandomForestRegressor**.
- Evaluation: MAE, MSE, R² score.
- Fairness classification: Based on prediction vs actual with ±30% tolerance.

---

## 🛠️ Installation & Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/house-price-fairness.git
```
## Dependencies:
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- google.colab (if running in Colab)

## 📊 Results
The Random Forest model achieved:
- Low MAE (better price prediction accuracy).
- Reasonable fairness classification with tolerance ±15–30%.
- Insights from EDA show significant differences in urban vs rural pricing patterns.

## 👥 Contributors
- S.P. WIdyasekara        – Preprocessing
- K.D.B.S. Senadeera      – Outlier removal & further preprocess
- B.D.F. Fernando         – Advanced EDA
- K.K.G.P.N. Samaraweera  – Model Training & Fairness Classification

## 📜 License
This project is licensed under the MIT License – free to use and modify.

