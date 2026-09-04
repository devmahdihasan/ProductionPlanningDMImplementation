# 🏭 Production Planning Using Data Mining

A Data Mining project that applies **Regression techniques** to production planning using synthetic, externally sourced, and user-uploaded datasets.

The system trains and compares multiple regression models, evaluates their performance, selects the best model based on **R² Score**, and provides an interactive **Streamlit application** for production prediction and custom regression analysis.

---

## ✨ Features

- Synthetic production dataset generation
- External production-planning dataset support
- Custom CSV dataset upload
- Dynamic target and feature selection
- Automatic numeric and categorical feature handling
- Missing-value preprocessing
- Data-quality and possible target-leakage warnings
- Three regression models
- Automatic model comparison
- Best model selection based on R² Score
- MAE, RMSE, and R² evaluation
- Separate trained models for built-in datasets
- Dynamic prediction interface for uploaded datasets
- Modular Streamlit interface
- Reusable machine-learning training pipeline

---

## 📊 Dataset Modes

The application provides three ways to work with regression data.

### 1. Synthetic Dataset

Synthetic production-planning data is generated locally using `generate_data.py`.

Each generated dataset contains **400 records** with the following variables:

| Feature         | Description                  |
| --------------- | ---------------------------- |
| `Demand`        | Expected product demand      |
| `Inventory`     | Current inventory            |
| `Workers`       | Number of workers            |
| `Working_Hours` | Working hours per day        |
| `Raw_Material`  | Available raw material       |
| `Production`    | Production quantity — target |

Generated datasets are stored inside:

```text
data/synthetic/
```

---

### 2. External Production Dataset

The project also supports the externally sourced:

**Multi-site Production-Distribution Prediction Dataset**

After preprocessing, the dataset contains **516 usable records** with production-planning variables including:

- Demand across multiple sites
- Inventory levels
- Production limits
- Manufacturing, storage, and shortage costs
- Vehicle capacity
- Inter-site product transfers

The built-in external model predicts:

```text
U1 — Production at Site 1
```

The dataset is stored inside:

```text
data/real/
```

---

### 3. Custom CSV Dataset

Users can upload their own CSV datasets directly through the Streamlit application.

The system allows the user to:

1. Upload a CSV file
2. Preview the dataset
3. Inspect column information
4. Select a numeric prediction target
5. Choose input features
6. Review data-quality warnings
7. Train three regression models
8. Compare model performance
9. Automatically select the best model
10. Enter new values and generate predictions

Common CSV formats using comma, semicolon, tab, or pipe separators are supported.

Numeric-like columns are automatically detected, while categorical variables can be encoded as part of the machine-learning pipeline.

---

## 🧠 Data Mining Technique

The selected Data Mining technique is:

**Regression**

Three regression algorithms are implemented:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression

Models are evaluated using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R² Score** — Coefficient of Determination

The model with the **highest R² Score** is selected as the best-performing model.

---

## 🏆 Built-in Model Results

### Synthetic Dataset

| Model                    |       MAE |      RMSE |         R² |
| ------------------------ | --------: | --------: | ---------: |
| **Linear Regression**    | **15.81** | **19.41** | **0.9974** |
| Decision Tree Regression |    103.58 |    135.93 |     0.8746 |
| Random Forest Regression |     60.29 |     78.98 |     0.9577 |

**Best Model by R²:** Linear Regression

The very high performance on the synthetic dataset is expected because its production target is generated from a controlled mathematical relationship between the input variables.

### External Dataset

| Model                    |       MAE |      RMSE |         R² |
| ------------------------ | --------: | --------: | ---------: |
| **Linear Regression**    | **73.20** | **93.48** | **0.4263** |
| Decision Tree Regression |     91.55 |    114.81 |     0.1346 |
| Random Forest Regression |     72.00 |     94.37 |     0.4153 |

**Best Model by R²:** Linear Regression

> Random Forest achieves a slightly lower MAE on the external dataset, while Linear Regression is selected because the project's model-selection criterion is the highest R² Score.

The difference between the synthetic and external results demonstrates how model performance can change when moving from controlled generated data to a more complex production-planning dataset.

---

## 🖥️ Streamlit Application

The application contains three interactive modes.

### Synthetic Dataset

Users enter:

- Expected demand
- Current inventory
- Number of workers
- Working hours
- Available raw material

The trained model predicts the expected **Production Quantity**.

### External Production Dataset

Users enter multi-site planning information including:

- Demand
- Inventory
- Production constraints
- Costs
- Vehicle capacity
- Inter-site distribution

The trained model predicts **Production at Site 1 (`U1`)**.

### Upload Custom CSV

Users can train regression models without modifying the source code.

After uploading a dataset, the application dynamically generates the required controls for:

- Target selection
- Feature selection
- Model training
- Performance comparison
- Numeric inputs
- Categorical inputs
- Prediction

This makes the project usable as a small general-purpose **regression experimentation interface** in addition to its primary production-planning use case.

---

## ⚠️ Data Quality Checks

For custom datasets, the application performs basic checks before model training and can warn about:

- Constant or empty features
- Features with high amounts of missing data
- ID/index-like columns
- High-cardinality categorical features
- Targets with very few unique values
- Extremely high feature-target correlations that may indicate target leakage

These warnings are advisory and do not automatically remove user-selected features.

---

## 🗂️ Project Structure

```text
ProductionPlanningDMImplementation/
│
├── components/
│   ├── __init__.py
│   ├── header.py
│   ├── footer.py
│   ├── dataset_selector.py
│   ├── synthetic_mode.py
│   ├── external_mode.py
│   └── custom_csv.py
│
├── ml/
│   ├── __init__.py
│   └── training.py
│
├── data/
│   ├── synthetic/
│   │   ├── production_data_001.csv
│   │   └── ...
│   │
│   └── real/
│       └── Multi-site Production-Distribution Prediction.csv
│
├── models/
│   ├── synthetic/
│   │   ├── best_model.pkl
│   │   ├── model_results.csv
│   │   └── metadata.pkl
│   │
│   └── real/
│       ├── best_model.pkl
│       ├── model_results.csv
│       └── metadata.pkl
│
├── app.py
├── generate_data.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- CSV

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/devmahdihasan/ProductionPlanningDMImplementation
cd ProductionPlanningDMImplementation
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 🚀 Usage

### Generate a synthetic dataset

```powershell
python generate_data.py
```

### Train the synthetic production model

```powershell
python train.py --source synthetic
```

### Train the external production model

```powershell
python train.py --source real
```

### Start the Streamlit application

```powershell
streamlit run app.py
```

The application will normally be available locally at:

```text
http://localhost:8501
```

---

## 🔄 Workflow

```text
                         ┌── Synthetic Dataset
                         │
Dataset Source ──────────┼── External Dataset
                         │
                         └── Custom CSV Upload
                                  │
                                  ▼
                         Data Preparation
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Regression Models     │
                     │                         │
                     │  • Linear Regression    │
                     │  • Decision Tree        │
                     │  • Random Forest        │
                     └────────────┬────────────┘
                                  │
                                  ▼
                         Model Evaluation
                       MAE • RMSE • R²
                                  │
                                  ▼
                     Best Model Selection
                         (Highest R²)
                                  │
                                  ▼
                            Prediction
```

---

## 📌 Notes

The synthetic dataset is generated specifically for demonstrating the Data Mining and regression workflow. Its highly predictable structure should not be interpreted as equivalent to real-world production performance.

The external dataset represents a more complex multi-site production-distribution planning problem and is processed separately rather than being artificially converted into the synthetic dataset structure.

The custom CSV mode extends the application beyond the built-in datasets by allowing users to experiment with their own regression problems.

---

## 📄 Purpose

This project was originally developed for academic and educational purposes as part of a **Data Mining Lab** project on **Production Planning**.

It has also been extended into a reusable regression application that demonstrates:

- Dataset preprocessing
- Regression model training
- Model comparison
- Performance evaluation
- Data-quality analysis
- Interactive machine-learning prediction

---

## 👨‍💻 Developer

Built by **devmahdihasan** · **NextGrid Digital**

---

⭐ If you find this project useful, consider starring the repository.
