import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)

from sklearn.svm import SVR

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/House Price Prediction Dataset.csv"
MODEL_PATH = "model/house_price_model.pkl"

TARGET_COLUMN = "Price"


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):

    st.error("❌ Dataset not found!")
    st.stop()

df = pd.read_csv(DATA_PATH)


# ============================================================
# CHECK TARGET COLUMN
# ============================================================

if TARGET_COLUMN not in df.columns:

    st.error(
        f"❌ Target column '{TARGET_COLUMN}' not found."
    )

    st.write(df.columns.tolist())

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🏠 House Price Prediction")

st.write(
    "Enter the house details below to predict the price."
)

st.divider()


# ============================================================
# PREPARE DATA
# ============================================================

X = df.drop(
    TARGET_COLUMN,
    axis=1
)

y = df[TARGET_COLUMN]


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_columns
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge":
        Ridge(alpha=1.0),

    "Lasso":
        Lasso(
            alpha=0.01,
            max_iter=10000
        ),

    "KNN":
        KNeighborsRegressor(
            n_neighbors=5
        ),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

    "SVR":
        SVR(),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        ),

    "AdaBoost":
        AdaBoostRegressor(
            n_estimators=100,
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=100,
            random_state=42
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

trained_models = {}

results = []


with st.spinner(
    "Preparing prediction model..."
):

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        y_pred = pipeline.predict(
            X_test
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        results.append(
            [
                name,
                r2,
                mae,
                rmse
            ]
        )

        trained_models[name] = pipeline


# ============================================================
# SELECT BEST MODEL INTERNALLY
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "R2 Score",
        "MAE",
        "RMSE"
    ]
)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
).reset_index(
    drop=True
)

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    best_model,
    MODEL_PATH
)


# ============================================================
# HOUSE INPUT
# ============================================================

st.subheader(
    "🏡 Enter House Details"
)

input_data = {}


for column in X.columns:

    # Categorical column

    if column in categorical_columns:

        options = (
            X[column]
            .dropna()
            .unique()
            .tolist()
        )

        input_data[column] = st.selectbox(
            column,
            options
        )

    # Numeric column

    elif column in numeric_columns:

        minimum = float(
            X[column].min()
        )

        maximum = float(
            X[column].max()
        )

        mean_value = float(
            X[column].mean()
        )

        input_data[column] = st.number_input(
            column,
            min_value=minimum,
            max_value=maximum,
            value=mean_value
        )


# ============================================================
# PREDICT
# ============================================================

st.divider()

if st.button(
    "🔮 Predict House Price",
    use_container_width=True
):

    try:

        input_df = pd.DataFrame(
            [input_data]
        )

        prediction = best_model.predict(
            input_df
        )

        predicted_price = prediction[0]

        st.success(
            f"🏠 Predicted House Price: ₹{predicted_price:,.2f}"
        )

    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )