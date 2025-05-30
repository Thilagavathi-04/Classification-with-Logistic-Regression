# 🧠 Logistic Regression - Binary Classification

This project demonstrates how to build a binary classification model using **Logistic Regression** on the **Breast Cancer Wisconsin Dataset** using Python, Pandas, Scikit-learn, and Matplotlib.

---

## 📌 Objective

Build a logistic regression classifier to predict whether a tumor is **malignant (0)** or **benign (1)** using patient medical measurements. The project also includes ROC analysis, threshold tuning, and sigmoid function visualization.

---

## 🧰 Tools and Libraries

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## 📊 Dataset

- Dataset: [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- Loaded via `sklearn.datasets.load_breast_cancer()`

---

## ✅ Workflow Steps

1. **Import libraries**  
   Essential data science and ML packages.

2. **Load & explore dataset**  
   - Convert the dataset into a DataFrame
   - Check dataset shape and preview the first few rows

3. **Train/Test split**  
   - Use `train_test_split` to divide data (80/20 ratio)

4. **Standardize features**  
   - Normalize features using `StandardScaler`

5. **Train Logistic Regression model**  
   - Fit `LogisticRegression` on training data

6. **Evaluate the model**  
   - Accuracy, confusion matrix, precision, recall, classification report

7. **Plot ROC curve and calculate AUC**  
   - Visualize performance of binary classification

8. **Threshold tuning**  
   - Change decision threshold (e.g., 0.3) and evaluate precision/recall

9. **Sigmoid Function Visualization**  
   - Plot the sigmoid function used in logistic regression

---

## 📈 Evaluation Metrics

- Accuracy
- Precision & Recall
- Confusion Matrix
- ROC Curve and AUC Score

---

## 📷 Sample Output

- ROC curve with AUC
- Sigmoid function curve

---


