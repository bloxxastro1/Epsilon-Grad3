import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Converted Streamlit App from Untitled-1.ipynb")

# --- Notebook Code Below ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

df = pd.read_csv("retail_store_sales.csv") # loading dataset

df.head() # initial data check``

df.describe() # statistical summary

df.info() # check for data types and null values

df = df.drop(columns = ["Customer ID","Transaction ID","Total Spent"]) # dropping irrelevant columns

df.drop_duplicates(inplace=True) # removing duplicates
df.duplicated().sum() # check for duplicates = 0


df['Price Per Unit'].fillna(df['Price Per Unit'].median(), inplace=True) # filling missing values with median
df['Quantity'].fillna(df['Quantity'].median(), inplace=True) # filling missing values with median

df.info() # check for data types and null values


df.info()

df['Transaction Date'] = pd.to_datetime(df['Transaction Date']) # converting to datetime
df['Month'] = df['Transaction Date'].dt.month # extracting month
df['Day'] = df['Transaction Date'].dt.day # extracting day
df['Year'] = df['Transaction Date'].dt.year # extracting year
df = df.drop(columns=['Transaction Date']) # dropping original date column
df['Total Spent'] = df['Price Per Unit'] * df['Quantity'] # creating total spent column
df.info() # check for data types


plt.pie(df["Category"].value_counts(), labels=df["Category"].value_counts().index, autopct='%1.1f%%') 
plt.title("Distribution of Item Categories") # pie chart of item categories
plt.show()

plt.pie(df['Payment Method'].value_counts(), labels=df['Payment Method'].value_counts().index, autopct='%1.1f%%')
plt.title("Distribution of Payment Methods") # pie chart of payment methods
plt.show()

plt.pie(df["Location"].value_counts(), labels=df["Location"].value_counts().index, autopct='%1.1f%%')
plt.title("Distribution of Store Locations") # pie chart of store locations
plt.show()

plt.scatter(df['Quantity'], df['Price Per Unit'])

sns.boxplot(x='Month', y='Quantity', data=df) # box plot of total spent by month
plt.title('Sales Distribution by Month') 
plt.show()

sns.boxplot(x='Day', y='Quantity', data=df) # box plot of total spent by Day
plt.title('Sales Distribution by Month') 
plt.show()

sns.boxplot(x='Year', y='Quantity', data=df) # box plot of total spent by month
plt.title('Sales Distribution by Month') 
plt.show()

df.loc[df["Discount Applied"].isnull(), "Discount Applied"] = np.nan

cols = ["Item","Payment Method", "Location", 'Discount Applied']
df= pd.get_dummies(df, columns=cols)

cols = ['Quantity', 'Price Per Unit']

lower_limit = df[cols].quantile(0.05)
upper_limit = df[cols].quantile(0.95)

df[cols] = df[cols].clip(lower=lower_limit, upper=upper_limit, axis=1)


df.head()


le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])
x = df.drop(columns=['Category'])
y = (df['Category'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # splitting data


models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DECT', DecisionTreeClassifier()))
models.append(('RC', RandomForestClassifier()))
models.append(('GUSS', GaussianNB()))  
models.append(('XG', XGBClassifier(use_label_encoder=False, eval_metric='logloss')))  
models.append(('SVM', SVC()))

for model in models:
    steps = [
        ('Scaler', MinMaxScaler()),
        ('Classifier', model[1])
    ]
    pipeline = Pipeline(steps=steps)
    
    scores = cross_validate(
        pipeline, x, y, cv=5,
        scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
        return_train_score=True
    )
    
    print(f"Model: {model[0]}")
    print("Train Accuracy : ", scores["train_accuracy"].mean())
    print("Test Accuracy  : ", scores["test_accuracy"].mean())
    print("Train Precision: ", scores["train_precision_weighted"].mean())
    print("Test Precision : ", scores["test_precision_weighted"].mean())
    print("Train Recall   : ", scores["train_recall_weighted"].mean())
    print("Test Recall    : ", scores["test_recall_weighted"].mean())
    print("Train F1 Score : ", scores["train_f1_weighted"].mean())
    print("Test F1 Score  : ", scores["test_f1_weighted"].mean())
    print('-' * 60)
