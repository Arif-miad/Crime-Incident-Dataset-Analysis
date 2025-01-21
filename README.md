# Crime Incident Dataset Analysis

## 📄 Dataset Overview

This dataset provides an in-depth view of criminal incidents, capturing details about offenders and victims, along with the nature and disposition of cases. It contains demographic details like age, gender, and race, enabling a comprehensive analysis of crime patterns, impacts, and trends.

### Columns:

1. **Disposition**: Current status of the case (e.g., Closed or Open).
2. **OffenderStatus**: Status of the offender (e.g., ARRESTED).
3. **Offender\_Race**: Race of the offender (e.g., BLACK, WHITE, ASIAN, etc.).
4. **Offender\_Gender**: Gender of the offender (MALE or FEMALE).
5. **Offender\_Age**: Age of the offender (numerical value).
6. **PersonType**: Role in the case (e.g., VICTIM, REPORTING PERSON).
7. **Victim\_Race**: Race of the victim (e.g., BLACK, WHITE, ASIAN, etc.).
8. **Victim\_Gender**: Gender of the victim (MALE or FEMALE).
9. **Victim\_Age**: Age of the victim (numerical value).
10. **Victim\_Fatal\_Status**: Indicates if the victim’s injuries were fatal or non-fatal.
11. **Report Type**: Type of report filed (e.g., Supplemental Report, Incident Report).
12. **Category**: Crime category (e.g., Theft, Vandalism, Violence, etc.).

---

## 💻 Project Workflow

This project involves the following key steps:

### 1. Exploratory Data Analysis (EDA):

- **Understand the data distribution**:
  - Use plots like histograms, pie charts, and count plots to analyze individual columns.
- **Relationships between features**:
  - Utilize pairplots, scatter plots, and heatmaps for insights.

### 2. Preprocessing:

- Handle missing values and clean the dataset.
- Encode categorical variables using **Label Encoding**.
- Apply feature scaling (MinMaxScaler) for numerical columns.

### 3. Data Splitting:

- Split the dataset into training and testing sets for machine learning models.

### 4. Model Development:

- Build and compare the performance of **Top 10 Classification Models**:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine
  - K-Nearest Neighbors
  - XGBoost
  - LightGBM
  - Naive Bayes
  - Neural Networks

### 5. Evaluation:

- Evaluate models using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.

---

## 📊 Key Visualizations

- **Univariate Analysis**:
  - Count plots for categorical columns (e.g., `Category`, `Disposition`).
  - Pie charts for gender and race distributions.
- **Bivariate Analysis**:
  - Scatter plots and line plots to understand relationships.
  - Heatmap to visualize feature correlations.
- **Advanced Plots**:
  - Box plots, violin plots, and KDE for detailed insights.

---

## 🛠 Tools & Libraries

- **Programming Language**: Python
- **Visualization Libraries**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/crime-dataset-analysis.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Crime_Analysis.ipynb
   ```

---

## 📈 Results

- Key findings from the analysis.
- Top-performing classification model and its metrics.

---

## 📬 Contact

Feel free to reach out through the following platforms:
<p align="center" style="background-color:#f7f7f7; color:#333; padding:20px; border-radius:10px; margin:20px 0;">
  <a href="https://www.kaggle.com/code/arifmia/comprehensive-analysis-and-classification-of-crimi" style="margin-right:10px; text-decoration:none; color:#0078d4;">
    📊 Kaggle Notebook
  </a>
  | 
  <a href="mailto:arifmiahcse@gmail.com" style="margin-right:10px; text-decoration:none; color:#d93025;">
    📧 Email
  </a>
  | 
  <a href="https://www.linkedin.com/in/arifmia" style="margin-right:10px; text-decoration:none; color:#0a66c2;">
    🌐 LinkedIn
  </a>
  | 
  <a href="https://wa.me/8801998246254" style="text-decoration:none; color:#25d366;">
    📱 WhatsApp
  </a>
</p>

---

### 🌟 Code Implementation

Below is an example of the code implementation used in this project:
### Import libraries
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
```
#### 1. Data Preprocessing
```python
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset
data = pd.read_csv('/kaggle/input/crime-data/crime_data.csv')

# Encode categorical variables
label_enc = LabelEncoder()
data['Offender_Race'] = label_enc.fit_transform(data['Offender_Race'])
data['Victim_Race'] = label_enc.fit_transform(data['Victim_Race'])
data['Category'] = label_enc.fit_transform(data['Category'])

# Split data
X = data.drop(columns=['Disposition'])
y = data['Disposition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### 2. Model Training Example
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

#### 3. Visualization Example
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
```

