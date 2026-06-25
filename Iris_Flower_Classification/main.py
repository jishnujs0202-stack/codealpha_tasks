# ==========================================================
# IRIS FLOWER CLASSIFICATION - PROFESSIONAL ML PROJECT
# ==========================================================

# =========================
# IMPORT LIBRARIES
# =========================
print("PROGRAM STARTED")
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("data/iris.csv")

print("\n==============================")
print("FIRST 5 ROWS")
print("==============================\n")

print(df.head())

# ==========================================================
# DATA UNDERSTANDING
# ==========================================================

print("\n==============================")
print("DATASET INFORMATION")
print("==============================\n")

print(df.info())

print("\n==============================")
print("MISSING VALUES")
print("==============================\n")

print(df.isnull().sum())

print("\n==============================")
print("STATISTICAL SUMMARY")
print("==============================\n")

print(df.describe())

# ==========================================================
# DATA VISUALIZATION
# ==========================================================

sns.set_style("whitegrid")

# Pairplot
sns.pairplot(df, hue="Species")
plt.suptitle("Iris Flower Pairplot", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))

numeric_df = df.drop(["Species", "Id"], axis=1)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=1
)

plt.title("Feature Correlation Heatmap")
plt.show()

# Boxplot
plt.figure(figsize=(10,6))

sns.boxplot(data=numeric_df)

plt.title("Feature Distribution")
plt.show()

# Species Count
plt.figure(figsize=(6,4))

sns.countplot(x='Species', data=df)

plt.title("Species Distribution")
plt.show()

# ==========================================================
# FEATURE & TARGET SELECTION
# ==========================================================

X = df.drop(["Species", "Id"], axis=1)
y = df["Species"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# MODEL BUILDING
# ==========================================================

models = {

    "Logistic Regression":
        LogisticRegression(),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "SVM":
        SVC(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

# ==========================================================
# MODEL TRAINING & EVALUATION
# ==========================================================

accuracy_scores = {}

for name, model in models.items():

    print("\n===================================")
    print(f"MODEL: {name}")
    print("===================================")

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    accuracy_scores[name] = accuracy

    print(f"\nAccuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

# ==========================================================
# ACCURACY COMPARISON
# ==========================================================

accuracy_df = pd.DataFrame({

    "Model": accuracy_scores.keys(),
    "Accuracy": accuracy_scores.values()

})

accuracy_df = accuracy_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================\n")

print(accuracy_df)

# Accuracy Plot
plt.figure(figsize=(10,5))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=accuracy_df
)

plt.title("Model Accuracy Comparison")
plt.ylim(0.9, 1.01)

plt.show()

# ==========================================================
# FINAL MODEL
# ==========================================================

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

final_model.fit(X_train, y_train)

# Save model
joblib.dump(final_model, "iris_model.pkl")

print("\nModel saved successfully!")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = final_model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({

    "Feature": feature_names,
    "Importance": importance

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df
)

plt.title("Feature Importance")
plt.show()

# ==========================================================
# SAMPLE PREDICTION
# ==========================================================

sample = [[5.1, 3.5, 1.4, 0.2]]

sample = scaler.transform(sample)

prediction = final_model.predict(sample)

print("\n==============================")
print("FINAL PREDICTION")
print("==============================\n")

print("Predicted Flower Species:", prediction[0])

# ==========================================================
# END
# ==========================================================