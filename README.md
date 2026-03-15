# Customer Churn Prediction using Machine Learning

## Project Overview

Customer churn is a major challenge for businesses because losing existing customers can significantly impact revenue. Acquiring new customers is usually more expensive than retaining current ones. Predicting which customers are likely to leave helps companies take proactive actions to improve customer retention.

In this project, a machine learning model is developed to predict whether a customer is likely to churn. The project demonstrates a complete machine learning workflow including data preprocessing, handling class imbalance, training a model, and evaluating its performance.

---

## Dataset

The dataset contains customer-related information that may influence churn behavior.

Example features include:

* Customer demographics
* Account information
* Service usage patterns
* Payment methods
* Contract type
* Customer tenure

### Target Variable

**Churn**

* 1 → Customer churned
* 0 → Customer stayed

The dataset is loaded and explored using the **pandas** library.

---

## Methodology

### Data Preprocessing

Before training the model, several preprocessing steps are applied:

* Handling missing values
* Encoding categorical variables
* Preparing features for the model
* Splitting the dataset into training and testing sets

These steps ensure the dataset is ready for machine learning models.

---

### Handling Class Imbalance

Customer churn datasets often have **class imbalance**, where the number of customers who stay is much larger than those who leave.

To handle this issue, the project uses:

**SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE helps balance the dataset by generating synthetic samples for the minority class.

---

### Machine Learning Model

The model used in this project is:

**Gradient Boosting Classifier**

Gradient Boosting is an ensemble learning technique that combines multiple weak learners to build a stronger predictive model. It works well for structured datasets such as customer churn data.

The model is trained after balancing the dataset using SMOTE.

---

## Technologies Used

* Python
* pandas
* numpy
* scikit-learn
* imbalanced-learn (SMOTE)
* seaborn
* matplotlib

---

## Model Evaluation

The model performance is evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score

A **confusion matrix** is also used to visualize prediction results.

---

## How to Run the Project

### 1. Clone the repository

git clone [https://github.com/yourusername/customer-churn-prediction.git](https://github.com/yourusername/customer-churn-prediction.git)

### 2. Navigate to the project directory

cd customer-churn-prediction

### 3. Install required libraries

pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib

### 4. Run the Python script

python churn_model.py

Make sure the dataset path in the script is correct.

---

## Results

The model learns patterns from customer data and predicts churn with reasonable performance.

Using SMOTE improves the model's ability to detect churn cases, which are usually the minority class in the dataset.

The results can be visualized using plots such as confusion matrix and other data analysis graphs.


---

## Limitations

Some limitations of this project include:

* Dataset size may be limited
* Feature engineering could be improved further
* Hyperparameter tuning is limited
* Additional validation is required before real-world deployment

---

## Future Improvements

Possible improvements include:

* Hyperparameter tuning for better performance
* Testing additional models such as Random Forest or XGBoost
* Feature importance analysis
* Deploying the model using Streamlit or Flask
* Creating a dashboard for churn prediction

---

## Author
Urooj Fatima

This project was developed as part of a machine learning practice project to explore customer churn prediction and build practical experience with data analysis and predictive modeling.

---






