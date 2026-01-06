import pandas as pd
import numpy as np
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# --------------------
# Feature definitions
# --------------------
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

CATEGORICAL_COLS = [
    "sex", "cp", "fbs", "restecg", "exang",
    "slope", "ca", "thal"
]

# --------------------
# Training function
# --------------------
def train(data_path="./data/processed/heart.csv"):
    df = pd.read_csv(data_path)
    df.replace("?", np.nan, inplace=True)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS)
    ])

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)
            probs = pipeline.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            roc = roc_auc_score(y_test, probs)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", roc)

            if model_name == "LogisticRegression":
                joblib.dump(pipeline, "heart_model_pipeline.joblib")
                mlflow.log_artifact("heart_model_pipeline.joblib")

if __name__ == "__main__":
    train()
