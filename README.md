# Customer Churn Prediction

## Project Overview

Customer Churn Prediction is a machine learning project that aims to
identify customers who are likely to stop using a company's services.

The project uses customer account information, service subscriptions,
contract details, and billing data to predict customer churn.

Three machine learning models were evaluated:
- Logistic Regression
- Decision Tree
- Random Forest

The models were compared using Accuracy, Precision, Recall, F1-Score,
and ROC-AUC.

## Dataset Description

The project uses the IBM Telco Customer Churn dataset.

- **Total records:** 7,043 customers
- **Original columns:** 21
- **Target variable:** `Churn`
- **Target classes:** `Yes` and `No`
- **Features:** Customer demographics, subscribed services,
  contract details, and billing information.

### Target Distribution

- **No Churn:** 5,174 customers
- **Churn:** 1,869 customers

The `customerID` column was removed because it is an identifier
rather than a useful predictive feature.
The `TotalCharges` column was converted to a numeric data type.

## Tools & Technologies

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Environment:** Google Colab
- **Model Persistence:** Joblib
- **Version Control:** Git and GitHub

## Project Workflow

1. **Data Loading:** Loaded the IBM Telco Customer Churn dataset.
2. **Data Cleaning:** Removed `customerID` and converted `TotalCharges`
   to a numeric format.
3. **Exploratory Data Analysis:** Examined churn distribution and
   customer features.
4. **Data Preprocessing:** Handled missing values, scaled numerical
   features, and encoded categorical features.
5. **Train-Test Split:** Split the data into training and testing sets
   using stratified sampling.
6. **Model Training:** Trained Logistic Regression, Decision Tree,
   and Random Forest models.
7. **Model Evaluation:** Compared models using Accuracy, Precision,
   Recall, F1-Score, and ROC-AUC.
8. **Threshold Tuning:** Selected a 0.40 probability threshold to
   improve churn detection.
9. **Model Pipeline:** Saved the preprocessing and trained model
   together using Joblib.
10. **Prediction Testing:** Loaded the saved artifact and tested
    predictions on customer records.

## Model Performance

Three machine learning models were evaluated on the test set.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 84.19% |
| Decision Tree | 79.28% | 63.85% | 50.53% | 56.42% | 83.59% |
| Random Forest | 76.79% | 54.75% | 72.46% | 62.37% | 83.97% |

### Threshold Tuning

The Random Forest model was evaluated using a probability threshold
of `0.40` instead of the default `0.50`.

| Metric | Result |
|---|---:|
| Accuracy | 74.17% |
| Precision | 50.83% |
| Recall | 81.55% |
| F1-Score | 62.63% |
| ROC-AUC | 83.97% |

The lower threshold improved Recall, helping the model identify more
customers who were likely to churn, at the cost of lower Precision
and more false-positive predictions.

**Note:** The threshold was selected using cross-validation on the
training data. Final performance was measured on the held-out test set.

## How to Use

### Requirements

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Load the Saved Model

```python
import joblib

artifact = joblib.load("customer_churn_model.joblib")
pipeline = artifact["pipeline"]
threshold = artifact["threshold"]
```

### Make a Prediction

Provide customer data in a Pandas DataFrame named `customer_data`.

```python
probability = pipeline.predict_proba(customer_data)[0, 1]
prediction = "Churn" if probability >= threshold else "No Churn"
print("Prediction:", prediction)
print("Churn Probability:", round(probability * 100, 2), "%")
```

The pipeline automatically applies preprocessing before prediction.

## Limitations

- The model was trained on a single Telco customer churn dataset.
- The selected threshold increases Recall but also produces more false positives.
- Model performance may vary on data from different companies.
- Predictions indicate churn risk and should not be treated as certain outcomes.

## Future Improvements

- Perform more extensive hyperparameter tuning.
- Explore additional classification models.
- Improve feature engineering.
- Evaluate the model on new, unseen datasets.
- Develop an interactive application for customer churn predictions.
