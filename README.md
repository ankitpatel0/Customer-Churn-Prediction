#  Customer Churn Prediction

Predicting customer churn in the telecom industry using machine learning techniques. This project analyzes customer data to identify those likely to discontinue services, allowing businesses to take preventive action and improve customer retention.

## 📁 Project Structure

Customer-Churn-Prediction/
├── Customer_Churn_Prediction_using_ML.ipynb # Jupyter Notebook with complete pipeline
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset
├── customer_churn_model.pkl # Trained model (saved using pickle)
├── encoders.pkl # Encoders used for categorical features
└── README.md # Project documentation


## 📌 Problem Statement

Churn prediction is crucial for telecom providers to understand customer behavior and improve service quality. Using machine learning, this project aims to classify whether a customer will churn (Yes/No) based on features such as services used, account tenure, charges, and demographics.

---

## 📊 Dataset Description

The dataset used is the **Telco Customer Churn** dataset from Kaggle:
- Rows: 7,043 customers
- Target: `Churn` (Yes/No)
- Features include:
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Account info: `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - Services: `PhoneService`, `InternetService`, `TechSupport`, etc.

---

## 🧰 Tech Stack

- **Language**: Python
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Tools**: Jupyter Notebook, Pickle (for model saving)

---

## 🛠️ Workflow

### 1. Data Preprocessing
- Missing values handling
- Categorical feature encoding (Label Encoding, One-Hot)
- Feature scaling using StandardScaler

### 2. Exploratory Data Analysis (EDA)
- Visualization of feature distributions
- Correlation heatmaps
- Churn distribution analysis

### 3. Model Building
- Splitting into training & test sets
- Model training (Logistic Regression, Random Forest, etc.)
- Evaluation using accuracy, precision, recall, F1-score

### 4. Model Serialization
- Saving model and encoders using `pickle` for deployment or reuse

---

## 🚀 How to Run

1. **Clone the Repository**:
```bash
git clone https://github.com/ankitpatel0/Customer-Churn-Prediction.git

cd Customer-Churn-Prediction

Install Dependencies:

bash
pip install -r requirements.txt

If requirements.txt is not present, install manually:

bash
pip install pandas numpy matplotlib seaborn scikit-learn

Launch Jupyter Notebook:

bash

jupyter notebook

Run the Notebook:
Open Customer_Churn_Prediction_using_ML.ipynb and run all cells step-by-step.

✅ Results
Update these values based on your actual results.

Accuracy: 80%

Precision: 75%

Recall: 70%

F1 Score: 72%

📦 Model Files
customer_churn_model.pkl – Trained classification model

encoders.pkl – Label/OneHot encoders for transforming categorical variables

These can be used for inference in a deployed application.

🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the model or add deployment support (e.g., using Flask or Streamlit).

📄 License
This project is licensed under the MIT License.

📬 Contact
Created by Ankit Patel
📧 ankitpatel531@gmail.com
🔗 GitHub Profile https://github.com/ankitpatel0
