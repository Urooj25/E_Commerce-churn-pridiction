# E_Commerce-churn-pridiction
# E-commerce Customer Satisfaction Prediction

💡 **Project Overview**

This project is focused on predicting **customer satisfaction levels** (Satisfied, Neutral, Unsatisfied) for an E-commerce business using **Machine Learning**. The goal is to help businesses understand customer behavior and take actionable decisions to improve customer experience.

---

## 🚀 Project Highlights

- **Objective:** Predict customer satisfaction based on behavior and purchase history.
- **Dataset:** Customer demographic & transactional data including Age, Gender, Total Spend, Items Purchased, Days Since Last Purchase, Discount Applied, City, and Membership Type.
- **Tools & Libraries:** Python, Pandas, Numpy, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib, Seaborn

---

## 🔧 Data Preprocessing & Feature Engineering

1. **Missing Values Handling:** Filled missing Age with median, Gender with mode.
2. **Encoding:** Label encoding for Gender, one-hot encoding for City & Membership Type.
3. **Feature Engineering:**  
   - `avg_order_value = Total Spend / (Items Purchased + 1)`  
   - `recency = Days Since Last Purchase`  
   - `Discount Applied` encoded as 0/1
4. **Scaling:** StandardScaler applied on numeric features.
5. **Handling Imbalance:** SMOTE used to balance class distribution.
6. **Removed Target Leakage:** Avoided features like `clv`, `monetary`, `frequency` that directly correlate with satisfaction.

---

## 🛠 Machine Learning Model

- **Model Used:** Random Forest Classifier
- **Hyperparameters:** Tuned for best performance (n_estimators=400, max_depth=12)
- **Evaluation:** Accuracy, Classification Report, Confusion Matrix
- **Visualization:** Feature Importance, Feature Correlation Heatmap

---

## 📊 Results

- **Model Accuracy:** 96% on test set ✅
- **Key Influencing Features:**  
  `recency`, `average order value`, `age`, `discount applied`
- **Visual Insights:** Feature correlation heatmap & importance chart
  

---

## 📈 Learning Outcomes

- Importance of **feature engineering** for improving model accuracy
- Handling **imbalanced datasets** using SMOTE
- Avoiding **target leakage** to prevent overfitting
- Translating raw E-commerce data into **actionable insights**
- Data visualization for **better understanding of features and model performance**

---





