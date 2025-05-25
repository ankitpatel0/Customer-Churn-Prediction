
# 📉 Customer Churn Prediction

## 📌 Overview

The Customer Churn Prediction project aims to identify customers who are likely to discontinue a service, enabling proactive retention strategies. By analyzing customer data and leveraging machine learning algorithms, the model predicts churn with high accuracy, assisting businesses in making informed decisions.

## 🚀 Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
- **Exploratory Data Analysis (EDA)**: Visualizing data distributions and relationships to uncover insights.
- **Model Training**: Implementing various machine learning algorithms to predict churn.
- **Model Evaluation**: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.
- **Model Deployment**: Saving the best-performing model for future predictions.

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - Joblib
- **Machine Learning Algorithms**:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)

## 📂 Installation & Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ankitpatel0/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook Customer_Churn_Prediction_using_ML.ipynb
   ```

4. **Make Predictions**:
   - Load the saved model `customer_churn_model.pkl`
   - Use the model to predict churn on new customer data

## 📁 Project Structure

```
Customer-Churn-Prediction/
├── Customer_Churn_Prediction_using_ML.ipynb  # Jupyter Notebook with the entire workflow
├── WA_Fn-UseC_-Telco-Customer-Churn.csv      # Dataset
├── customer_churn_model.pkl                  # Saved machine learning model
├── encoders.pkl                              # Saved encoders for categorical variables
├── requirements.txt                          # List of dependencies
└── README.md                                 # Project documentation
```

## 📈 Evaluation Metrics

- **Accuracy**: Percentage of correct predictions.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual positives that were identified correctly.
- **F1-Score**: Harmonic mean of precision and recall.

*Note: Specific metric values can be found in the Jupyter Notebook `Customer_Churn_Prediction_using_ML.ipynb`.*

## 🤝 Contribution

Contributions are welcome! If you have suggestions for improvements or want to contribute to the project, please fork the repository and submit a pull request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📬 Contact

- **Name**: Ankit Patel
- **GitHub**: [ankitpatel0](https://github.com/ankitpatel0)
