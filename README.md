# Placement Prediction using Logistic Regression

## 📌 Project Overview

This is a beginner-friendly Machine Learning project that predicts whether a student will get placed or not based on:

* CGPA
* IQ

The project uses **Logistic Regression** for classification.

---

# 🚀 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Mlxtend
* Pickle

---

# 📂 Dataset Information

The dataset contains:

| Column    | Description                                   |
| --------- | --------------------------------------------- |
| cgpa      | Student CGPA                                  |
| iq        | Student IQ                                    |
| placement | Placement Status (0 = Not Placed, 1 = Placed) |

---

# 📊 Data Visualization

Scatter plot was used to visualize the data.

```python
plt.scatter(df['cgpa'], df['iq'], c=df['placement'])
```

---

# ⚙️ Project Workflow

## 1. Import Libraries

```python
import numpy as np
import pandas as pd
```

---

## 2. Load Dataset

```python
df = pd.read_csv('placement.csv')
```

---

## 3. Data Preprocessing

```python
df = df.iloc[:,1:]
```

---

## 4. Feature Selection

```python
X = df.iloc[:,0:2]
Y = df.iloc[:,-1]
```

---

## 5. Train-Test Split

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2
)
```

---

## 6. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

---

## 7. Model Training

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(x_train, y_train)
```

---

## 8. Prediction

```python
y_pred = clf.predict(x_test)
```

---

## 9. Accuracy Evaluation

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

---

## 10. Decision Boundary Visualization

```python
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(x_train, y_train.values, clf=clf)
```

---

## 11. Save Model

```python
import pickle

pickle.dump(clf, open('model.pkl', 'wb'))
```

---

# 📈 Model Performance

* Algorithm Used: Logistic Regression
* Accuracy: 50%

> Accuracy may vary because the dataset is very small.

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend
```

---

# ▶️ Run Project

Run Jupyter Notebook:

```bash
jupyter notebook
```

---

# 📁 Project Structure

```text
placement-prediction/
│
├── placement.csv
├── model.pkl
├── notebook.ipynb
├── README.md
└── requirements.txt
```

---

# 🧠 Concepts Learned

* Data preprocessing
* Feature scaling
* Logistic Regression
* Model evaluation
* Decision boundary plotting
* Model saving using Pickle

---

# 📌 Future Improvements

* Use larger dataset
* Improve model accuracy
* Add frontend using Flask or Streamlit
* Deploy project online

---

# 👨‍💻 Author

Ritikrai7

---

# ⭐ If you like this project

Give this repository a star ⭐
