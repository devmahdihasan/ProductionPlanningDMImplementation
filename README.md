# 🏭 Production Planning Using Data Mining and Regression

A Data Mining Lab project that applies regression techniques to production planning and provides an interactive Streamlit interface for predicting production quantity.

## 📌 Project Overview

Production planning is the process of determining an appropriate production quantity based on factors such as expected demand, current inventory, workforce, working hours, and available raw materials.

This project uses Data Mining and Machine Learning to learn relationships between these production-related factors and predict the expected production quantity.

The project implements three regression models:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression

The models are evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

The best-performing model is then used by a Streamlit web interface to generate production predictions from user input.

---

## 🎯 Objectives

- Understand the application of Data Mining in Production Planning.
- Prepare and analyze production-related data.
- Apply Regression as a Data Mining technique.
- Train and compare at least three regression models.
- Evaluate model performance using MAE, RMSE, and R².
- Select the best-performing model.
- Save the trained model for future predictions.
- Build an interactive prediction interface using Streamlit.

---

## 📊 Dataset

The project uses a synthetic production planning dataset containing **300 records**.

### Features

| Column | Description | Type |
|---|---|---|
| `Demand` | Expected demand for the product | Numerical |
| `Inventory` | Current available inventory | Numerical |
| `Workers` | Number of workers available | Numerical |
| `Working_Hours` | Working hours per day | Numerical |
| `Raw_Material` | Available raw material | Numerical |
| `Production` | Production quantity to be predicted | Numerical / Target |

### Input Features

```text
Demand
Inventory
Workers
Working_Hours
Raw_Material
```

### Target

```text
Production
```

> **Note:** The dataset is synthetic and is intended for academic/lab demonstration of the Data Mining workflow. A real-world production planning system would use historical manufacturing data.

---

## 🧠 Data Mining Technique

### Regression

Regression is a supervised machine learning technique used to predict a continuous numerical value.

In this project, regression is used to predict:

```text
Production
```

from:

```text
Demand
Inventory
Workers
Working_Hours
Raw_Material
```

Regression is appropriate because production quantity is a numerical target.

---

## 🤖 Machine Learning Models

### 1. Linear Regression

Linear Regression models the relationship between the input features and production using a linear equation.

Conceptually:

```text
Production =
β₀ + β₁(Demand)
   + β₂(Inventory)
   + β₃(Workers)
   + β₄(Working_Hours)
   + β₅(Raw_Material)
```

### 2. Decision Tree Regression

Decision Tree Regression predicts the target by recursively splitting the data according to feature values.

Configuration used:

```text
max_depth = 6
```

### 3. Random Forest Regression

Random Forest Regression combines multiple decision trees to produce a more robust prediction.

Configuration used:

```text
Number of Trees = 100
max_depth = 8
```

---

## 🔄 Data Processing Workflow

The project follows this workflow:

```text
Production Dataset
       ↓
Load CSV
       ↓
Separate Features and Target
       ↓
Train/Test Split
       ↓
Train 3 Regression Models
       ↓
Evaluate Models
       ↓
Compare Performance
       ↓
Select Best Model
       ↓
Save Best Model
       ↓
Streamlit Prediction Interface
```

The dataset was divided into:

| Dataset | Records | Percentage |
|---|---:|---:|
| Training | 240 | 80% |
| Testing | 60 | 20% |
| Total | 300 | 100% |

---

## 📈 Model Evaluation

Three metrics were used.

### MAE — Mean Absolute Error

MAE measures the average absolute difference between actual and predicted values.

```text
MAE = Average |Actual - Predicted|
```

**Lower is better.**

### RMSE — Root Mean Squared Error

RMSE measures prediction error while giving more weight to larger errors.

```text
RMSE = √(Average((Actual - Predicted)²))
```

**Lower is better.**

### R² Score

R² measures how well the model explains variation in the target variable.

**Higher is better.**

---

## 🏆 Model Performance

The exact results obtained from the project are:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Linear Regression** | **19.16** | **24.55** | **0.9844** |
| Decision Tree Regression | 72.15 | 93.46 | 0.7740 |
| Random Forest Regression | 43.59 | 53.05 | 0.9272 |

### Best Model

**Linear Regression**

```text
R²   = 0.9844
MAE  = 19.16
RMSE = 24.55
```

Linear Regression achieved the highest R² and the lowest MAE and RMSE among the three models, so it was selected as the final prediction model.

---

## 🖥️ Streamlit Interface

The project includes an interactive Streamlit web application.

The user can enter:

- Expected Demand
- Current Inventory
- Number of Workers
- Working Hours per Day
- Available Raw Material

The application then uses the trained Linear Regression model to predict production quantity.

### Example Input

```text
Expected Demand       = 1000
Current Inventory     = 100
Number of Workers     = 30
Working Hours/Day     = 9
Available Raw Material = 1100
```

### Example Output

```text
Predicted Production: 1,133 units
```

---

## 🗂️ Project Structure

```text
ProductionPlanningDMImplementation/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   └── production_data.csv
│
├── models/
│   ├── best_model.pkl
│   └── model_results.csv
│
├── app.py
├── train.py
└── generate_data.py
```

### File Descriptions

#### `generate_data.py`

Generates the production planning dataset.

#### `train.py`

Responsible for:

- Loading the dataset.
- Preparing features and target.
- Splitting the data.
- Training the three models.
- Evaluating model performance.
- Selecting the best model.
- Saving the best model.
- Saving model comparison results.

#### `app.py`

Runs the Streamlit web application and performs production predictions.

#### `data/production_data.csv`

Contains the production planning dataset.

#### `models/best_model.pkl`

Contains the saved trained Linear Regression model.

#### `models/model_results.csv`

Contains the performance results of the three models.

#### `requirements.txt`

Contains the Python dependencies required to run the project.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning and evaluation |
| Joblib | Saving/loading trained model |
| Streamlit | Interactive web interface |
| CSV | Dataset storage |
| VS Code | Development environment |
| Git/GitHub | Version control |

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd ProductionPlanningDMImplementation
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```powershell
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 🧪 Training the Models

If the model files need to be regenerated:

```powershell
python train.py
```

This generates/updates:

```text
models/best_model.pkl
models/model_results.csv
```

If the dataset needs to be regenerated:

```powershell
python generate_data.py
```

---

## 🔐 Git Configuration

The virtual environment should not be committed to Git.

The `.gitignore` file contains entries such as:

```gitignore
.venv/
__pycache__/
*.pyc
```

This keeps unnecessary Python environment files out of the repository.

---

## 🧩 System Architecture

```text
                 PRODUCTION PLANNING
                         │
                         ▼
              ┌─────────────────────┐
              │ Production Dataset  │
              │     300 Records     │
              └──────────┬──────────┘
                         │
                         ▼
                 Data Preparation
                         │
                         ▼
                    80% / 20%
                 Train / Test Split
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
       Linear Reg.   Decision Tree  Random Forest
            │            │            │
            └────────────┼────────────┘
                         ▼
                  Model Evaluation
                         │
                         ▼
               🏆 Linear Regression
                  R² = 0.9844
                         │
                         ▼
                  Saved Model
                best_model.pkl
                         │
                         ▼
                Streamlit Interface
                         │
                         ▼
                   User Input
                         │
                         ▼
               Production Prediction
```

---

## ✅ Advantages

- Data-driven production prediction.
- Multiple machine learning models are compared.
- Simple and interactive user interface.
- Fast prediction after model training.
- Easy-to-understand regression workflow.
- Trained model can be reused without retraining.

---

## ⚠️ Limitations

- The dataset is synthetic.
- Only five input features are currently used.
- The application does not use real-time factory data.
- The model's performance depends on the quality of the training data.
- Real production planning may require additional factors such as machine availability, production cost, machine downtime, product type, and seasonal demand.

---

## 🚀 Future Scope

Possible future improvements include:

- Using real manufacturing datasets.
- Adding production cost.
- Adding machine availability.
- Adding machine downtime.
- Adding product type.
- Adding historical production trends.
- Adding seasonal demand.
- Connecting the application to a database.
- Using real-time production data.
- Testing additional machine learning algorithms.
- Deploying the application online.
- Adding production analytics dashboards.

---

## 📋 Lab Demonstration

For the lab demonstration, the following input can be used:

```text
Demand              = 1000
Inventory           = 100
Workers             = 30
Working Hours       = 9
Raw Material        = 1100
```

Expected demonstration output:

```text
Predicted Production: 1,133 units
```

The model comparison should also show:

```text
Linear Regression       R² = 0.9844
Decision Tree Regression R² = 0.7740
Random Forest Regression R² = 0.9272
```

---

## 🎓 Viva Quick Answers

### What is Production Planning?

Production planning is the process of deciding how much product should be produced based on factors such as demand, inventory, workforce, working hours, and raw materials.

### Why did you use Regression?

Because the target variable, Production, is a continuous numerical value.

### Which Data Mining technique did you use?

Regression.

### What are your input features?

Demand, Inventory, Workers, Working Hours, and Raw Material.

### What is your target variable?

Production.

### Which models did you implement?

Linear Regression, Decision Tree Regression, and Random Forest Regression.

### Which model performed best?

Linear Regression.

### What was its R² score?

0.9844.

### What was its MAE?

19.16.

### What was its RMSE?

24.55.

### Why was Linear Regression selected?

It achieved the highest R² and the lowest MAE and RMSE among the three models.

### What is the purpose of Streamlit?

It provides an interactive interface where users can enter production-related information and receive a prediction.

### Is the dataset real?

No. It is a synthetic dataset created for academic/lab demonstration. A real-world system would use historical manufacturing data.

---

## 📌 Conclusion

This project demonstrates an end-to-end application of Data Mining to Production Planning. A synthetic dataset containing 300 records was used to train and compare three regression models. Linear Regression achieved the best performance with an R² score of 0.9844, MAE of 19.16, and RMSE of 24.55. The trained model was integrated into a Streamlit application that allows users to provide production-related information and obtain a predicted production quantity.

The project demonstrates how Data Mining can support data-driven production planning and provides a foundation that could be extended using real-world manufacturing data.

---

## 👨‍💻 Author

**Name:** __________________________  
**Student ID:** _____________________  
**Course:** Data Mining Lab  
**Semester:** 7th Semester  
**Project:** Production Planning Using Data Mining and Regression

---

## 📄 License

This project was developed for academic and educational purposes as part of a Data Mining Lab course.
