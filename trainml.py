# trainml.py
import pandas as pd
import numpy as np
from evaluation import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load dataset safely
df_advanced = pd.read_csv(
    "df_advanced.csv",
    engine="python",
    sep=",",
    on_bad_lines="skip"
)

# Split into features and target
X = df_advanced.drop(columns=['Fraudulent'])
y = df_advanced['Fraudulent']

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X.select_dtypes(include=[np.number]), y,
                                                    test_size=0.3, stratify=y, random_state=42)

# Define models once
models = {
    "Logistic_Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42),
    "Gradient_Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.9, class_weight="balanced", random_state=42)
}

# Evaluate
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    results[name] = evaluate_model(model, X_train, y_train, X_test, y_test, recall_target=0.8)
