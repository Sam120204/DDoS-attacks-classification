# DDoS Attack Detection Using Machine Learning

## Introduction
Distributed Denial of Service (DDoS) attacks are significant threats to the stability and reliability of online services. Detecting and mitigating these attacks is crucial to maintaining the integrity of networks and services. This project focuses on classifying DDoS attacks using various machine learning models. The dataset used for this project is the IDS 2017 dataset, which is publicly available and provides a comprehensive set of features for detecting DDoS attacks.

The project involves several key steps: data preprocessing, exploration, splitting, model training, evaluation, and comparison. Each step is crucial to building an effective DDoS detection model. We employ multiple machine learning algorithms, including Random Forest, Logistic Regression, and Neural Networks, to classify the attacks and evaluate their performance using various metrics.

## Table of Contents
1. [Importing Libraries](#1-importing-libraries)
2. [Data Pre-processing](#2-data-pre-processing)
3. [Data Exploring](#3-data-exploring)
4. [Data Splitting](#4-data-splitting)
5. [Model Training](#5-model-training)
    - [Random Forest](#random-forest)
    - [Logistic Regression](#logistic-regression)
    - [Neural Network](#neural-network)
6. [Model Evaluation](#6-model-evaluation)
    - [Accuracy](#accuracy)
    - [F1 Score](#f1-score)
    - [Recall](#recall)
    - [Precision](#precision)
    - [Confusion Matrix](#confusion-matrix)
7. [Model Comparison](#7-model-comparison)

## 1. Importing Libraries
This chapter covers the importation of essential libraries used for data manipulation, visualization, model training, and evaluation. Libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn are utilized.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, auc
```

## 2. Data Pre-processing
This section involves preparing the data for analysis by cleaning and transforming it. Steps include handling missing values, converting categorical labels to numerical values, and ensuring data types are appropriate for analysis.

```python
# Example code for data pre-processing
df = pd.read_csv("DDoS.csv")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
# Convert categorical labels to numerical values if necessary
df['Label'] = df['Label'].map({'BENIGN': 0, 'DDoS': 1})
```

## 3. Data Exploring
Data exploration involves generating descriptive statistics and visualizations to understand the distribution and relationships within the dataset. This step helps in identifying important features and potential issues with the data.

```python
# Example code for data exploration
print(df.describe())
sns.pairplot(df)
plt.show()
```

## 4. Data Splitting
In this chapter, the data is split into training and testing sets. This step is crucial for evaluating the model's performance on unseen data, ensuring that the model generalizes well.

```python
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 5. Model Training
This section covers the training of different machine learning models:

### Random Forest
An ensemble method that uses multiple decision trees to improve predictive accuracy.

```python
# Random Forest
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
```

### Logistic Regression
A statistical model used for binary classification.

```python
# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
```

### Neural Network
A computational model inspired by the human brain, capable of capturing complex patterns in the data.

```python
# Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)
```

## 6. Model Evaluation
The trained models are evaluated using various metrics:

### Accuracy
The proportion of correctly predicted instances.

```python
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Neural Network Accuracy:", accuracy_score(y_test, nn_pred))
```

### F1 Score
The harmonic mean of precision and recall, useful for imbalanced datasets.

```python
print("Random Forest F1 Score:", f1_score(y_test, rf_pred))
print("Logistic Regression F1 Score:", f1_score(y_test, lr_pred))
print("Neural Network F1 Score:", f1_score(y_test, nn_pred))
```

### Recall
The ability of the model to identify all relevant instances.

```python
print("Random Forest Recall:", recall_score(y_test, rf_pred))
print("Logistic Regression Recall:", recall_score(y_test, lr_pred))
print("Neural Network Recall:", recall_score(y_test, nn_pred))
```

### Precision
The accuracy of the positive predictions.

```python
print("Random Forest Precision:", precision_score(y_test, rf_pred))
print("Logistic Regression Precision:", precision_score(y_test, lr_pred))
print("Neural Network Precision:", precision_score(y_test, nn_pred))
```

### Confusion Matrix
A table that describes the performance of the classification model.

```python
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Neural Network Confusion Matrix:\n", confusion_matrix(y_test, nn_pred))
```

## 7. Model Comparison
In this chapter, the performance of the different models is compared using ROC curves and AUC scores. This comparison helps in identifying the best-performing model for DDoS attack classification.

```python
# Example code for ROC curve and AUC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_model.predict_proba(X_test)[:,1])
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_model.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(rf_fpr, rf_tpr, label='Random Forest (AUC = %0.2f)' % auc(rf_fpr, rf_tpr))
plt.plot(lr_fpr, lr_tpr, label='Logistic Regression (AUC = %0.2f)' % auc(lr_fpr, lr_tpr))
plt.plot(nn_fpr, nn_tpr, label='Neural Network (AUC = %0.2f)' % auc(nn_fpr, nn_tpr))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
```

## Conclusion
The project systematically addresses the detection and classification of DDoS attacks using multiple machine learning models. By following the structured approach outlined in the chapters, we aim to build a robust model that can effectively distinguish between benign and malicious network traffic.
```
