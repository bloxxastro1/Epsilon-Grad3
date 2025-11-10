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
import warnings
warnings.filterwarnings('ignore')

st.title("Grad Project Epsilon - Retail Store Sales Analysis")

# --- Data Loading & Preprocessing ---
st.header("Data Loading & Preprocessing")

try:
    github_url = "https://raw.githubusercontent.com/bloxxastro1/Epsilon-Grad3/main/retail_store_sales.csv"
    df = pd.read_csv(github_url, sep=",", engine="python", on_bad_lines="skip")
    st.success("✅ Data loaded successfully!")
    st.write(f"Dataset shape: {df.shape}")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

st.subheader("Initial Data Preview")
st.write(df.head())

# Data cleaning
columns_to_drop = ['Customer ID', 'Transaction ID', 'Total Spent']
existing_cols = [col for col in columns_to_drop if col in df.columns]
df = df.drop(existing_cols, axis=1)
st.write(f"✅ Dropped columns: {existing_cols}")

# Remove duplicates
initial_rows = len(df)
df.drop_duplicates(inplace=True)
final_rows = len(df)
st.write(f"✅ Removed {initial_rows - final_rows} duplicate rows")

# Handle missing values
st.write("🔍 Handling missing values...")
missing_before = df[['Price Per Unit', 'Quantity']].isnull().sum()
df['Price Per Unit'].fillna(df['Price Per Unit'].median(), inplace=True)
df['Quantity'].fillna(df['Quantity'].median(), inplace=True)
missing_after = df[['Price Per Unit', 'Quantity']].isnull().sum()
st.write(f"✅ Filled missing values - Before: {missing_before}, After: {missing_after}")

# Handle Discount Applied
if 'Discount Applied' in df.columns:
    df['Discount Applied'].fillna('No Discount', inplace=True)
    st.write("✅ Handled Discount Applied column")

# Date processing
st.write("📅 Processing date features...")
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
df['Month'] = df['Transaction Date'].dt.month
df['Day'] = df['Transaction Date'].dt.day
df['Year'] = df['Transaction Date'].dt.year
df = df.drop(columns=['Transaction Date'])
df['Total Spent'] = df['Price Per Unit'] * df['Quantity']
st.write("✅ Date features processed")

st.subheader("Data Info After Preprocessing")
st.write(df.info())

# --- Data Visualization ---
st.header("📊 Data Distribution Visualizations")

# Calculate value counts for pie charts
category_counts = df["Category"].value_counts()
payment_counts = df['Payment Method'].value_counts()
location_counts = df["Location"].value_counts()

# Create columns for better layout
st.subheader("Pie Charts")
col1, col2, col3 = st.columns(3)

with col1:
    fig1 = px.pie(values=category_counts.values, 
                  names=category_counts.index, 
                  title='Item Categories')
    st.plotly_chart(fig1, use_container_width=True, key="pie_category")

with col2:
    fig2 = px.pie(values=payment_counts.values, 
                  names=payment_counts.index, 
                  title='Payment Methods')
    st.plotly_chart(fig2, use_container_width=True, key="pie_payment")

with col3:
    fig3 = px.pie(values=location_counts.values, 
                  names=location_counts.index, 
                  title='Store Locations')
    st.plotly_chart(fig3, use_container_width=True, key="pie_location")

# Scatter plot
st.subheader("Scatter Plot")
fig4 = px.scatter(df, x='Quantity', y='Price Per Unit', 
                 title='Quantity vs Price Per Unit')
st.plotly_chart(fig4, use_container_width=True, key="scatter_quantity_price")

# Box plots
st.subheader("Box Plots")
col4, col5, col6 = st.columns(3)

with col4:
    fig5 = px.box(df, x='Month', y='Quantity', 
                 title='Sales by Month')
    st.plotly_chart(fig5, use_container_width=True, key="box_month")

with col5:
    fig6 = px.box(df, x='Day', y='Quantity', 
                 title='Sales by Day')
    st.plotly_chart(fig6, use_container_width=True, key="box_day")

with col6:
    fig7 = px.box(df, x='Year', y='Quantity', 
                 title='Sales by Year')
    st.plotly_chart(fig7, use_container_width=True, key="box_year")

# --- Feature Engineering ---
st.header("🔧 Feature Engineering")

# One-hot encoding
st.write("🔄 Applying one-hot encoding...")
cols_to_encode = ["Item", "Payment Method", "Location", "Discount Applied"]
cols_to_encode = [col for col in cols_to_encode if col in df.columns]

st.write(f"Encoding columns: {cols_to_encode}")
df = pd.get_dummies(df, columns=cols_to_encode)
st.write(f"✅ One-hot encoding completed. New shape: {df.shape}")

# Outlier handling
st.write("📊 Handling outliers...")
numerical_cols = ['Quantity', 'Price Per Unit']
for col in numerical_cols:
    if col in df.columns:
        lower_limit = df[col].quantile(0.05)
        upper_limit = df[col].quantile(0.95)
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        st.write(f"✅ Clipped outliers for {col}")

st.subheader("Data After Feature Engineering")
st.write(f"Final dataset shape: {df.shape}")
st.write(df.head())

# --- Machine Learning Models ---
st.header("🤖 Machine Learning Model Comparison")

# Check if we have enough data
if len(df) < 100:
    st.warning("⚠️ Dataset might be too small for reliable ML results")
    
# Encode target variable
st.write("🎯 Encoding target variable...")
le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])
st.write(f"Target classes: {len(le.classes_)}")

# Prepare features and target
x = df.drop(columns=['Category'])
y = df['Category']

st.write(f"Features shape: {x.shape}")
st.write(f"Target shape: {y.shape}")

# Check for any remaining NaN values
if x.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    st.warning("⚠️ There are still NaN values in the data. Handling them...")
    x = x.fillna(x.median())
    y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
st.write(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Define models with simpler parameters for faster execution
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('Gaussian Naive Bayes', GaussianNB()),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('SVM', SVC(random_state=42))
]

# Progress tracking
progress_bar = st.progress(0)
status_text = st.empty()

# Train and evaluate models
results = []
for i, (name, model) in enumerate(models):
    progress = i / len(models)
    progress_bar.progress(progress)
    status_text.text(f"Training {name}...")
    
    with st.expander(f"Model: {name}", expanded=False):
        try:
            st.write(f"🔄 Training {name}...")
            
            # Create pipeline
            steps = [
                ('Scaler', MinMaxScaler()),
                ('Classifier', model)
            ]
            pipeline = Pipeline(steps=steps)
            
            # Cross-validation with fewer folds for speed
            scores = cross_validate(
                pipeline, x, y, cv=3,  # Reduced from 5 to 3 for speed
                scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                return_train_score=True,
                n_jobs=-1  # Use parallel processing
            )
            
            # Store results
            result = {
                'Model': name,
                'Train_Accuracy': scores['train_accuracy'].mean(),
                'Test_Accuracy': scores['test_accuracy'].mean(),
                'Train_Precision': scores['train_precision_weighted'].mean(),
                'Test_Precision': scores['test_precision_weighted'].mean(),
                'Train_Recall': scores['train_recall_weighted'].mean(),
                'Test_Recall': scores['test_recall_weighted'].mean(),
                'Train_F1': scores['train_f1_weighted'].mean(),
                'Test_F1': scores['test_f1_weighted'].mean()
            }
            results.append(result)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Scores")
                st.metric("Accuracy", f"{scores['train_accuracy'].mean():.4f}")
                st.metric("Precision", f"{scores['train_precision_weighted'].mean():.4f}")
                st.metric("Recall", f"{scores['train_recall_weighted'].mean():.4f}")
                st.metric("F1 Score", f"{scores['train_f1_weighted'].mean():.4f}")
            
            with col2:
                st.subheader("Test Scores")
                st.metric("Accuracy", f"{scores['test_accuracy'].mean():.4f}")
                st.metric("Precision", f"{scores['test_precision_weighted'].mean():.4f}")
                st.metric("Recall", f"{scores['test_recall_weighted'].mean():.4f}")
                st.metric("F1 Score", f"{scores['test_f1_weighted'].mean():.4f}")
                
            st.success(f"✅ {name} completed successfully!")
                
        except Exception as e:
            st.error(f"❌ Error with {name}: {str(e)}")
            # Continue with next model instead of stopping

# Update progress bar
progress_bar.progress(1.0)
status_text.text("Model training completed!")

# Display summary results
if results:
    st.header("📈 Model Comparison Summary")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.style.highlight_max(axis=0, subset=[col for col in results_df.columns if col != 'Model']))
    
    # Best model
    best_model_idx = results_df['Test_Accuracy'].idxmax()
    best_model = results_df.loc[best_model_idx]
    st.success(f"🏆 Best Model: {best_model['Model']} with Test Accuracy: {best_model['Test_Accuracy']:.4f}")
else:
    st.error("❌ No models were successfully trained. Check the errors above.")

st.success("🎉 Analysis completed!")
