import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as px

st.title("Grad Project Epsilon - Retail Store Sales Analysis")

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

github_url = "https://raw.githubusercontent.com/bloxxastro1/Epsilon-Grad3/main/retail_store_sales.csv"
df = pd.read_csv(github_url, sep=",", engine="python", on_bad_lines="skip")
 # loading dataset

st.write(df.head()) # initial data check``

columns_to_drop = ['Customer ID', 'Transaction ID', 'Total Spent']
existing_cols = [col for col in columns_to_drop if col in df.columns]
df = df.drop(existing_cols, axis=1) # dropping irrelevant columns

df.drop_duplicates(inplace=True) # removing duplicates
df.duplicated().sum() # check for duplicates = 0


df['Price Per Unit'].fillna(df['Price Per Unit'].median(), inplace=True) # filling missing values with median
df['Quantity'].fillna(df['Quantity'].median(), inplace=True) # filling missing values with median

df['Transaction Date'] = pd.to_datetime(df['Transaction Date']) # converting to datetime
df['Month'] = df['Transaction Date'].dt.month # extracting month
df['Day'] = df['Transaction Date'].dt.day # extracting day
df['Year'] = df['Transaction Date'].dt.year # extracting year
df = df.drop(columns=['Transaction Date']) # dropping original date column
df['Total Spent'] = df['Price Per Unit'] * df['Quantity'] # creating total spent column
st.write(df.info()) # check for data types

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Pie chart for Category distribution
category_counts = df["Category"].value_counts()
fig1 = px.pie(values=category_counts.values, 
              names=category_counts.index, 
              title='Distribution of Item Categories')
st.plotly_chart(fig1, use_container_width=True)

# Pie chart for Payment Method distribution
payment_counts = df['Payment Method'].value_counts()
fig2 = px.pie(values=payment_counts.values, 
              names=payment_counts.index, 
              title='Distribution of Payment Methods')
st.plotly_chart(fig2, use_container_width=True)

# Pie chart for Location distribution
location_counts = df["Location"].value_counts()
fig3 = px.pie(values=location_counts.values, 
              names=location_counts.index, 
              title='Distribution of Store Locations')
st.plotly_chart(fig3, use_container_width=True)

# Scatter plot for Quantity vs Price Per Unit
fig4 = px.scatter(df, x='Quantity', y='Price Per Unit', 
                 title='Quantity vs Price Per Unit')
st.plotly_chart(fig4, use_container_width=True)

# Box plot for Sales Distribution by Month
fig5 = px.box(df, x='Month', y='Quantity', 
             title='Sales Distribution by Month')
st.plotly_chart(fig5, use_container_width=True)

# Box plot for Sales Distribution by Day
fig6 = px.box(df, x='Day', y='Quantity', 
             title='Sales Distribution by Day')
st.plotly_chart(fig6, use_container_width=True)

# Box plot for Sales Distribution by Year
fig7 = px.box(df, x='Year', y='Quantity', 
             title='Sales Distribution by Year')
st.plotly_chart(fig7, use_container_width=True)

# Full width for scatter plot
fig4 = px.scatter(df, x='Quantity', y='Price Per Unit', 
                 title='Quantity vs Price Per Unit')
st.plotly_chart(fig4, use_container_width=True)

# Box plots in columns
col4, col5, col6 = st.columns(3)

with col4:
    fig5 = px.box(df, x='Month', y='Quantity', title='By Month')
    st.plotly_chart(fig5, use_container_width=True)

with col5:
    fig6 = px.box(df, x='Day', y='Quantity', title='By Day')
    st.plotly_chart(fig6, use_container_width=True)

with col6:
    fig7 = px.box(df, x='Year', y='Quantity', title='By Year')
    st.plotly_chart(fig7, use_container_width=True)

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
    
    st.write(f"Model: {model[0]}")
    st.write("Train Accuracy : ", scores["train_accuracy"].mean())
    st.write("Test Accuracy  : ", scores["test_accuracy"].mean())
    st.write("Train Precision: ", scores["train_precision_weighted"].mean())
    st.write("Test Precision : ", scores["test_precision_weighted"].mean())
    st.write("Train Recall   : ", scores["train_recall_weighted"].mean())
    st.write("Test Recall    : ", scores["test_recall_weighted"].mean())
    st.write("Train F1 Score : ", scores["train_f1_weighted"].mean())
    st.write("Test F1 Score  : ", scores["test_f1_weighted"].mean())
    st.write('-' * 60)
