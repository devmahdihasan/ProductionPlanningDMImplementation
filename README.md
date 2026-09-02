# 🏭 Production Planning Using Data Mining

A Data Mining project that applies **Regression techniques** to Production Planning using both synthetic and externally sourced production data.

The system trains and compares three regression models, selects the best-performing model, and provides an interactive **Streamlit interface** for production prediction.

---

## ✨ Features

- Synthetic production dataset generation
- External production-planning dataset support
- Three regression models
- Automatic model comparison
- Best model selection
- Separate trained models for each dataset source
- MAE, RMSE, and R² evaluation
- Interactive Streamlit prediction interface
- Synthetic and external prediction modes

---

## 📊 Dataset Sources

### 1. Synthetic Dataset

Generated locally using `generate_data.py`.

Each generated dataset contains **400 records** with:

| Feature | Description |
|---|---|
| `Demand` | Expected product demand |
| `Inventory` | Current inventory |
| `Workers` | Number of workers |
| `Working_Hours` | Working hours per day |
| `Raw_Material` | Available raw material |
| `Production` | Production quantity — target |

New datasets are automatically stored inside:

```text
data/synthetic/
```

### 2. External Production Dataset

The project also supports the externally sourced:

**Multi-site Production-Distribution Prediction Dataset**

After preprocessing, the dataset contains **516 usable records** covering production-planning information such as:

- Demand across multiple sites
- Inventory
- Production limits
- Manufacturing, storage, and shortage costs
- Distribution capacity
- Inter-site transfers

The external model predicts **Site 1 production (`U1`)**.

The dataset is stored inside:

```text
data/real/
```

---

## 🧠 Data Mining Technique

The selected Data Mining technique is:

**Regression**

Three regression algorithms are compared:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression

The models are evaluated using:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² Score

---

## 🏆 Model Results

### Synthetic Dataset

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Linear Regression** | **15.81** | **19.41** | **0.9974** |
| Decision Tree Regression | 103.58 | 135.93 | 0.8746 |
| Random Forest Regression | 60.29 | 78.98 | 0.9577 |

**Best Model:** Linear Regression

### External Dataset

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Linear Regression** | **73.20** | **93.48** | **0.4263** |
| Decision Tree Regression | 91.55 | 114.81 | 0.1346 |
| Random Forest Regression | 72.00 | 94.37 | 0.4153 |

**Best Model:** Linear Regression

> Model performance differs significantly between the controlled synthetic dataset and the more complex external production-planning dataset.

---

## 🖥️ Streamlit Application

The application provides two prediction modes:

### Synthetic Dataset

Users provide:

- Expected Demand
- Current Inventory
- Number of Workers
- Working Hours
- Available Raw Material

The model predicts the expected **Production Quantity**.

### External Production Dataset

Users provide multi-site information including:

- Demand
- Inventory
- Production constraints
- Costs and capacity
- Inter-site distribution

The model predicts **Production at Site 1**.

---

## 🗂️ Project Structure

```text
ProductionPlanningDMImplementation/
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
- Joblib
- Streamlit
- CSV

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd ProductionPlanningDMImplementation
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate it

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

### Train using synthetic data

```powershell
python train.py --source synthetic
```

### Train using external data

```powershell
python train.py --source real
```

### Start the application

```powershell
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🔄 Workflow

```text
Synthetic Data ─────┐
                    │
                    ├──► Regression Models
                    │          │
External Data ──────┘          │
                               ▼
                        Model Evaluation
                               │
                               ▼
                         Best Model
                               │
                               ▼
                       Streamlit App
                               │
                               ▼
                    Production Prediction
```

---

## 📌 Notes

The synthetic dataset is generated specifically for demonstrating the Data Mining workflow.

The external dataset represents a more complex multi-site production-distribution planning problem and is processed separately rather than being artificially converted into the synthetic dataset structure.

---

## 📄 Purpose

This project was developed for academic and educational purposes as part of a **Data Mining Lab** project on **Production Planning**.