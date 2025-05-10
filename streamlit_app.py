import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- Page Setup ---
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("📈 Sales Prediction & Analysis")
st.markdown("**MS2I | Machine Learning Project 2024/2025**")

# --- Sidebar ---
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Choose Model", ("Linear Regression", "Random Forest"))

if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
    max_depth = st.sidebar.slider("Max Depth", 1, 30, 10)

scale_data = st.sidebar.checkbox("Apply Feature Scaling (StandardScaler)")

# --- File Upload ---
st.subheader("1. Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV with 'Date' and 'Weekly_Sales'", type=["csv"])

# --- Helper Functions ---
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Weekly_Sales'])
    df['Day'] = (df['Date'] - df['Date'].min()).dt.days
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    return df

def prepare_data(df, features, scale=False):
    X = df[features]
    y = df['Weekly_Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

def plot_predictions(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6, color='teal')
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales")
    st.pyplot(fig)

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette="viridis", ax=ax)
    ax.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig)

def plot_correlation(df, features):
    corr = df[features + ['Weekly_Sales']].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# --- Main Workflow ---
if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)

    if 'Weekly_Sales' not in df.columns:
        st.error("Dataset must contain 'Weekly_Sales' column.")
    else:
        st.success("✅ Data loaded successfully!")

        st.subheader("2. Data Overview")
        st.dataframe(df.head())
        st.write("📊 Basic Statistics")
        st.dataframe(df.describe())

        st.subheader("3. Feature Correlation")
        default_features = [col for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag', 'Day'] if col in df.columns]
        if default_features:
            plot_correlation(df, default_features)

        st.subheader("4. Feature Selection")
        selected_features = st.multiselect(
            "Select Features for Prediction",
            options=df.columns.drop(['Weekly_Sales', 'Date']),
            default=default_features
        )

        if selected_features:
            X_train, X_test, y_train, y_test = prepare_data(df, selected_features, scale=scale_data)

            # Train and evaluate
            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

            mse, r2, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

            st.subheader("5. Model Evaluation")
            st.write(f"📉 **Mean Squared Error (MSE)**: `{mse:.2f}`")
            st.write(f"🔎 **R² Score**: `{r2:.2f}`")

            st.subheader("6. Prediction Visualization")
            plot_predictions(y_test, y_pred)

            if model_choice == "Random Forest":
                st.subheader("7. Feature Importance")
                plot_feature_importance(model, selected_features)

            # Optional: export predictions
            if st.checkbox("Export Predictions as CSV"):
                result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                st.download_button("Download Predictions", result_df.to_csv(index=False), file_name="predictions.csv")
