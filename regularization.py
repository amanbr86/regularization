import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ======================================================
# CREATE IMAGES FOLDER
# ======================================================

os.makedirs("images", exist_ok=True)

# ======================================================
# LOAD DATASET
# ======================================================

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================================
# FEATURE SCALING
# ======================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# BEFORE REGULARIZATION
# ======================================================

print("\n===== BEFORE REGULARIZATION =====")

before_model = LogisticRegression(
    C=10000,
    max_iter=10000
)

before_model.fit(X_train, y_train)

y_train_before = before_model.predict(X_train)
y_test_before = before_model.predict(X_test)

train_acc_before = accuracy_score(
    y_train,
    y_train_before
)

test_acc_before = accuracy_score(
    y_test,
    y_test_before
)

print("Training Accuracy:", train_acc_before)
print("Testing Accuracy:", test_acc_before)

# ======================================================
# CONFUSION MATRIX BEFORE
# ======================================================

cm_before = confusion_matrix(
    y_test,
    y_test_before
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_before,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix Before Regularization")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "images/confusion_before.png"
)

plt.close()

# ======================================================
# L1 REGULARIZATION
# ======================================================

print("\n===== L1 REGULARIZATION =====")

l1_model = LogisticRegression(
    solver='liblinear',
    l1_ratio=1.0,
    C=0.1,
    max_iter=10000
)

l1_model.fit(X_train, y_train)

y_train_l1 = l1_model.predict(X_train)
y_test_l1 = l1_model.predict(X_test)

train_acc_l1 = accuracy_score(
    y_train,
    y_train_l1
)

test_acc_l1 = accuracy_score(
    y_test,
    y_test_l1
)

print("Training Accuracy:", train_acc_l1)
print("Testing Accuracy:", test_acc_l1)

# ======================================================
# CONFUSION MATRIX L1
# ======================================================

cm_l1 = confusion_matrix(
    y_test,
    y_test_l1
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_l1,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix - L1")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "images/confusion_l1.png"
)

plt.close()

# ======================================================
# L2 REGULARIZATION
# ======================================================

print("\n===== L2 REGULARIZATION =====")

l2_model = LogisticRegression(
    l1_ratio=0.0,
    C=0.1,
    max_iter=10000
)

l2_model.fit(X_train, y_train)

y_train_l2 = l2_model.predict(X_train)
y_test_l2 = l2_model.predict(X_test)

train_acc_l2 = accuracy_score(
    y_train,
    y_train_l2
)

test_acc_l2 = accuracy_score(
    y_test,
    y_test_l2
)

print("Training Accuracy:", train_acc_l2)
print("Testing Accuracy:", test_acc_l2)

# ======================================================
# CONFUSION MATRIX L2
# ======================================================

cm_l2 = confusion_matrix(
    y_test,
    y_test_l2
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_l2,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix - L2")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "images/confusion_l2.png"
)

plt.close()

# ======================================================
# ACCURACY COMPARISON GRAPH
# ======================================================

models = [
    "Before Reg",
    "L1",
    "L2"
]

train_scores = [
    train_acc_before,
    train_acc_l1,
    train_acc_l2
]

test_scores = [
    test_acc_before,
    test_acc_l1,
    test_acc_l2
]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10,6))

plt.bar(
    x - width/2,
    train_scores,
    width,
    label='Train Accuracy'
)

plt.bar(
    x + width/2,
    test_scores,
    width,
    label='Test Accuracy'
)

plt.xticks(x, models)

plt.ylabel("Accuracy")

plt.title(
    "Accuracy Comparison"
)

plt.legend()

plt.savefig(
    "images/accuracy_comparison.png"
)

plt.close()

# ======================================================
# COEFFICIENT VISUALIZATION
# ======================================================

plt.figure(figsize=(12,6))

plt.plot(
    l1_model.coef_[0],
    label='L1 Coefficients'
)

plt.plot(
    l2_model.coef_[0],
    label='L2 Coefficients'
)

plt.axhline(
    0,
    color='black'
)

plt.legend()

plt.title(
    "Coefficient Comparison"
)

plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")

plt.savefig(
    "images/coefficients.png"
)

plt.close()

# ======================================================
# CLASSIFICATION REPORTS
# ======================================================

print("\n===== CLASSIFICATION REPORT BEFORE =====")
print(
    classification_report(
        y_test,
        y_test_before
    )
)

print("\n===== CLASSIFICATION REPORT L1 =====")
print(
    classification_report(
        y_test,
        y_test_l1
    )
)

print("\n===== CLASSIFICATION REPORT L2 =====")
print(
    classification_report(
        y_test,
        y_test_l2
    )
)

print("\nALL VISUALIZATIONS SAVED INSIDE IMAGES FOLDER")