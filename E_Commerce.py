# ====================================
# 1️⃣ IMPORT LIBRARIES
# ====================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

plt.ion()  # plots interactive

# ====================================
# 2️⃣ LOAD DATA
# ====================================
df = pd.read_csv(r"C:\Users\Mr Laptop\Downloads\My_Python_Projects\E-commerce Customer Behavior - Sheet1.csv")

# Fill missing
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df = df.dropna(subset=['Satisfaction Level'])

df['Discount Applied'] = df['Discount Applied'].astype(int)


# ====================================
# 3️⃣ ENCODING
# ====================================
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['City','Membership Type'], drop_first=True)

# ====================================
# 4️⃣ FEATURE ENGINEERING (NO LEAKAGE)
# ====================================
# Only meaningful features
df['avg_order_value'] = df['Total Spend'] / (df['Items Purchased']+1)
df['recency'] = df['Days Since Last Purchase']
df['age'] = df['Age']

X = df[['Gender','age','avg_order_value','recency','Discount Applied'] +
       [c for c in df.columns if 'City_' in c or 'Membership Type_' in c]]
y = df['Satisfaction Level']

# ====================================
# 5️⃣ TRAIN TEST SPLIT
# ====================================
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    stratify=y,
                                                    random_state=42)

# ====================================
# 6️⃣ HANDLE IMBALANCE
# ====================================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ====================================
# 7️⃣ FEATURE SCALING
# ====================================
numeric_cols = X_train_res.select_dtypes(include=['int64','float64']).columns
scaler = StandardScaler()
X_train_res[numeric_cols] = scaler.fit_transform(X_train_res[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric features
numeric_features = X_train_res.select_dtypes(include=['int64','float64']).columns

# Compute correlation
corr_matrix = X_train_res[numeric_features].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()

# Save high-resolution image for LinkedIn
plt.savefig("feature_correlation_heatmap.png", dpi=300)
plt.show()

# ====================================
# 8️⃣ RANDOM FOREST
# ====================================
rf = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42)
rf.fit(X_train_res, y_train_res)

# ====================================
# 9️⃣ PREDICTIONS
# ====================================
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy (Realistic):", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ====================================
# 🔟 CONFUSION MATRIX
# ====================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show(block=True)

# ====================================
# 1️⃣1️⃣ FEATURE IMPORTANCE
# ====================================
importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
importance = importance.sort_values(by='Importance', ascending=False)
print("\nTop Features:\n", importance.head(10))

plt.figure(figsize=(8,6))
sns.barplot(x=importance['Importance'][:10], y=importance['Feature'][:10])
plt.title("Top 10 Feature Importance")
plt.tight_layout()
plt.show(block=True)
