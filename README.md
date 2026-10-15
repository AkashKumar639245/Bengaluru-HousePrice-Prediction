# 🏠 Bengaluru House Price Prediction

A machine learning project that predicts house prices in Bengaluru, India, from location, total area (sq ft), number of bedrooms (BHK) and bathrooms. It covers the full workflow: data cleaning, feature engineering, outlier removal, one-hot encoding and a Linear Regression model evaluated with K-Fold style cross-validation.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Project Structure](#-project-structure)
4. [Tech Stack](#-tech-stack)
5. [Installation & Usage](#-Installation-Usage)
6. [Methodology](#-methodology)
7. [Data Size at Each Stage](#-data-size-at-each-stage)
8. [Results](#-results)
9. [Key Learnings](#-key-learnings)
10. [Limitations & Future Work](#-limitations--future-work)
11. [Author](#-author)

---

## 📖 Project Overview

Real estate prices in Bengaluru vary a lot with location, size and configuration. The raw data is messy: missing values, area given as ranges or in different units, inconsistent location names and many unrealistic listings.

This project:
- Cleans and standardises the raw housing data
- Engineers useful features (`bhk`, `price_per_sqft`)
- Removes outliers using domain logic and statistics
- Encodes the location feature with one-hot encoding
- Trains a Linear Regression model and validates it with cross-validation

---

## 📂 Dataset

- **File:** `bengaluru_house_prices.csv`
- **Size:** 13,320 rows × 9 columns
- **Target variable:** `price` (in **lakh INR**)

| Column | Description |
|---|---|
| `area_type` | Type of area (Super built-up, Built-up, Plot, Carpet) |
| `availability` | Possession date or "Ready To Move" |
| `location` | Locality in Bengaluru |
| `size` | Configuration, e.g. `2 BHK`, `4 Bedroom` |
| `society` | Society name |
| `total_sqft` | Total area in square feet (may be a range or other unit) |
| `bath` | Number of bathrooms |
| `balcony` | Number of balconies |
| `price` | Price in lakh INR (target) |

> The dataset is the popular *Bengaluru House Price Data* set (commonly found on Kaggle). Place the CSV in the project root before running the notebook.

---

## 🗂 Project Structure

```
bengaluru-house-price-prediction/
│
├── project2.ipynb                 # Main notebook (EDA, cleaning, modelling)
├── bengaluru_house_prices.csv     # Raw dataset (add this file yourself)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🛠 Tech Stack

- **Language:** Python 3
- **Data handling:** Pandas, NumPy
- **Visualisation:** Matplotlib
- **Machine learning:** Scikit-learn (`LinearRegression`, `train_test_split`, `ShuffleSplit`, `cross_val_score`)
- **Environment:** Jupyter Notebook

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter and open the notebook
jupyter notebook
```

Make sure `bengaluru_house_prices.csv` is in the same folder as the notebook, then run all cells in order.

**`requirements.txt`**
```
pandas
numpy
matplotlib
scikit-learn
jupyter
```

---

## 🔬 Methodology

### 1. Data Loading & Exploration
- Loaded the CSV with Pandas and checked its shape and `area_type` distribution.

### 2. Data Cleaning
- Dropped columns that add little predictive value: `area_type`, `society`, `balcony`, `availability`.
- Dropped rows with missing values (`location`, `size`, `bath`).

### 3. Feature Engineering
- **`bhk`**: extracted the number from `size` (e.g. `"2 BHK"` and `"4 Bedroom"` → `2`, `4`).
- **`total_sqft` cleaning**: some values were ranges (`"2100 - 2850"`) or used other units (`Sq. Meter`, `Perch`). Ranges were converted to their average; values that could not be converted were set to `NaN`.
- **`price_per_sqft`**: `price × 100000 / total_sqft`, used for outlier detection.

### 4. Dimensionality Reduction (Location)
- Stripped whitespace from location names (1,304 → 1,293 unique locations).
- Locations with **10 or fewer listings** were grouped into an `"other"` category, reducing unique locations to **242**.

### 5. Outlier Removal
| Step | Technique |
|---|---|
| Unrealistic space per room | Removed listings where `total_sqft / bhk < 300` (e.g. 6 bedrooms in 1,020 sq ft) |
| Price per sq ft outliers | Per location, kept only values within **one standard deviation** of the mean |
| BHK price anomalies | Per location, removed N BHK homes whose price per sq ft was lower than the mean of (N−1) BHK homes |
| Bathroom anomalies | Removed listings where `bath >= bhk + 2` |

Scatter plots (2 BHK vs 3 BHK for locations like Rajaji Nagar and Hebbal) and histograms (price per sq ft, bathrooms) were used to inspect the data before and after cleaning.

### 6. Encoding
- One-hot encoded `location` using `pd.get_dummies`, dropping the `other` column to avoid the dummy-variable trap.
- Final feature matrix: **244 features** (`total_sqft`, `bath`, `bhk` + 241 location columns).

### 7. Model Training
- **Split:** 80% train / 20% test (`random_state=10`)
- **Model:** `LinearRegression`

### 8. Model Validation
- **5-fold ShuffleSplit cross-validation** (`test_size=0.2`, `random_state=0`) to check that performance is stable across different splits.

---

## 📉 Data Size at Each Stage

| Stage | Rows |
|---|---|
| Raw data | 13,320 |
| After dropping missing values | 13,246 |
| After removing `total_sqft / bhk < 300` | 12,502 |
| After price-per-sqft outlier removal | 10,241 |
| After BHK outlier removal | 7,329 |
| After bathroom outlier removal (final) | **7,251** |

---

## 🏆 Results

| Metric | Score |
|---|---|
| R² on test set (80/20 split) | **0.845** |
| Cross-validation R² (5 folds) | 0.824, 0.772, 0.851, 0.808, 0.837 |
| Mean cross-validation R² | **≈ 0.82** |

The model explains about 82–85% of the variance in house prices, and the cross-validation scores are consistent across folds.

---

## 💡 Key Learnings

- Real-world data needs heavy cleaning before modelling; unit and format inconsistencies matter.
- Domain knowledge (such as a minimum sq ft per bedroom) is a powerful outlier filter.
- Grouping rare categories reduces the number of one-hot features and noise.
- Cross-validation gives a more reliable estimate than a single train/test split.

---

## ⚠️ Limitations & Future Work

- Only Linear Regression is used so far; other models (Lasso, Decision Tree, Random Forest, XGBoost) could be compared.
- Hyperparameter tuning (e.g. `GridSearchCV`) is not yet included.
- The trained model is not yet exported or served; a `predict_price(location, sqft, bath, bhk)` function, a saved model (`pickle`/`joblib`) and a Flask/Streamlit web app would make it usable.
- Rows with unconvertible area units (Sq. Meter, Perch, etc.) are dropped rather than converted.
- Prices are from a single snapshot in time and may not reflect current market conditions.

---


---

⭐ If you found this project useful, consider giving it a star!
