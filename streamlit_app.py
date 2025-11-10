import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

st.title("Grad Project Epsilon - Retail Store Sales Analysis")

# --- Data Loading & Preprocessing ---
github_url = "https://raw.githubusercontent.com/bloxxastro1/Epsilon-Grad3/main/retail_store_sales.csv"
df = pd.read_csv(github_url, sep=",", engine="python", on_bad_lines="skip")

st.subheader("Initial Data Preview")
st.write(df.head())

# Data cleaning
columns_to_drop = ['Customer ID', 'Transaction ID', 'Total Spent']
existing_cols = [col for col in columns_to_drop if col in df.columns]
df = df.drop(existing_cols, axis=1)

df.drop_duplicates(inplace=True)
st.write(f"Duplicates found: {df.duplicated().sum()}")

# Handle missing values
df['Price Per Unit'].fillna(df['Price Per Unit'].median(), inplace=True)
df['Quantity'].fillna(df['Quantity'].median(), inplace=True)

# Handle Discount Applied - fix the NaN assignment
if 'Discount Applied' in df.columns:
    df['Discount Applied'].fillna('No Discount', inplace=True)

# Date processing
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
df['Month'] = df['Transaction Date'].dt.month
df['Day'] = df['Transaction Date'].dt.day
df['Year'] = df['Transaction Date'].dt.year
df = df.drop(columns=['Transaction Date'])
df['Total Spent'] = df['Price Per Unit'] * df['Quantity']

st.subheader("Data Info After Preprocessing")
st.write(df.info())

# --- Data Visualization ---
st.header("Data Distribution Visualizations")

# Calculate value counts for pie charts
category_counts = df["Category"].value_counts()
payment_counts = df['Payment Method'].value_counts()
location_counts = df["Location"].value_counts()

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    fig1 = px.pie(values=category_counts.values, 
                  names=category_counts.index, 
                  title='Distribution of Item Categories')
    st.plotly_chart(fig1, use_container_width=True, key="pie_category")

with col2:
    fig2 = px.pie(values=payment_counts.values, 
                  names=payment_counts.index, 
                  title='Distribution of Payment Methods')
    st.plotly_chart(fig2, use_container_width=True, key="pie_payment")

with col3:
    fig3 = px.pie(values=location_counts.values, 
                  names=location_counts.index, 
                  title='Distribution of Store Locations')
    st.plotly_chart(fig3, use_container_width=True, key="pie_location")

# Scatter plot
fig4 = px.scatter(df, x='Quantity', y='Price Per Unit', 
                 title='Quantity vs Price Per Unit')
st.plotly_chart(fig4, use_container_width=True, key="scatter_quantity_price")

# Box plots in columns
col4, col5, col6 = st.columns(3)

with col4:
    fig5 = px.box(df, x='Month', y='Quantity', 
                 title='Sales Distribution by Month')
    st.plotly_chart(fig5, use_container_width=True, key="box_month")

with col5:
    fig6 = px.box(df, x='Day', y='Quantity', 
                 title='Sales Distribution by Day')
    st.plotly_chart(fig6, use_container_width=True, key="box_day")

with col6:
    fig7 = px.box(df, x='Year', y='Quantity', 
                 title='Sales Distribution by Year')
    st.plotly_chart(fig7, use_container_width=True, key="box_year")

# --- Feature Engineering ---
st.header("Feature Engineering")

# One-hot encoding
cols_to_encode = ["Item", "Payment Method", "Location", "Discount Applied"]
cols_to_encode = [col for col in cols_to_encode if col in df.columns]

df = pd.get_dummies(df, columns=cols_to_encode)

# Outlier handling for numerical columns
numerical_cols = ['Quantity', 'Price Per Unit']
for col in numerical_cols:
    if col in df.columns:
        lower_limit = df[col].quantile(0.05)
        upper_limit = df[col].quantile(0.95)
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

st.subheader("Data After Feature Engineering")
st.write(df.head())

# --- Machine Learning Models ---
st.header("Machine Learning Model Comparison")

# Encode target variable
le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])

# Prepare features and target
x = df.drop(columns=['Category'])
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Gaussian Naive Bayes', GaussianNB()),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('SVM', SVC())
]

# Train and evaluate models
for name, model in models:
    with st.expander(f"Model: {name}"):
        try:
            steps = [
                ('Scaler', MinMaxScaler()),
                ('Classifier', model)
            ]
            pipeline = Pipeline(steps=steps)
            
            scores = cross_validate(
                pipeline, x, y, cv=5,
                scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                return_train_score=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Train Accuracy", f"{scores['train_accuracy'].mean():.4f}")
                st.metric("Train Precision", f"{scores['train_precision_weighted'].mean():.4f}")
                st.metric("Train Recall", f"{scores['train_recall_weighted'].mean():.4f}")
                st.metric("Train F1 Score", f"{scores['train_f1_weighted'].mean():.4f}")
            
            with col2:
                st.metric("Test Accuracy", f"{scores['test_accuracy'].mean():.4f}")
                st.metric("Test Precision", f"{scores['test_precision_weighted'].mean():.4f}")
                st.metric("Test Recall", f"{scores['test_recall_weighted'].mean():.4f}")
                st.metric("Test F1 Score", f"{scores['test_f1_weighted'].mean():.4f}")
                
        except Exception as e:
            st.error(f"Error with {name}: {str(e)}")

st.success("Model comparison completed!")
