# ❤️ CardioPredict AI – Heart Disease Prediction

🔗 **Live Demo:** [Open the app](https://heart-disease-prediction-mzjavhevfgdovbnenhyvzy.streamlit.app/)

A machine learning web app that predicts whether a patient is likely to have heart disease from basic clinical information. Built with **Python, Scikit-learn and Streamlit**.

## 📌 Overview

- Trained and compared **7 classification models** on the Heart Failure Prediction dataset (918 patients, 11 features).
- The **K-Nearest Neighbors (KNN)** model is used in the final app.
- A clean dark-themed **Streamlit** interface lets a user enter patient details and get a prediction with class probabilities.

## 📊 Model Results

| Model | Accuracy |
|---|---|
| Gradient Boosting | 87.50% |
| Logistic Regression | 86.96% |
| **KNN (used in app)** | **86.41%** |
| Random Forest | 85.87% |
| SVM | 84.78% |
| Naive Bayes | 84.78% |
| Decision Tree | 79.89% |

Evaluation: 80/20 train-test split (`random_state=42`).

## 🔄 Workflow

1. **Data cleaning** – checked missing values and duplicates; replaced invalid `0` values in `Cholesterol` and `RestingBP` with the mean of valid values.
2. **EDA** – distribution plots for Age, RestingBP, Cholesterol and MaxHR.
3. **Encoding** – one-hot encoding of categorical features (`pd.get_dummies`).
4. **Scaling** – `StandardScaler` on features.
5. **Modeling** – trained and compared 7 algorithms.
6. **Deployment** – saved model, scaler and column order with `joblib` and built a Streamlit app.

## 🗂️ Project Structure

```
heart-disease-prediction/
├── app.py                          # Streamlit web app
├── Heart_Disease_Prediction.ipynb  # Data analysis & model training
├── KNN_heart.pkl                   # Trained KNN model
├── scaler.pkl                      # Fitted StandardScaler
├── columns.pkl                     # Feature column order
├── requirements.txt
└── README.md
```

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/younis-ai/heart-disease-prediction.git
cd heart-disease-prediction

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 🧾 Input Features

Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Maximum Heart Rate, Exercise-Induced Angina, Oldpeak, ST Slope.

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Joblib · Streamlit

## 📁 Dataset

[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle, by fedesoriano).

## 👤 Author

**Younis** – [GitHub](https://github.com/younis-ai)
