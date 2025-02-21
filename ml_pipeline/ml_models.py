import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load data
def load_data(df, test_size, target):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Feature Engineering Pipeline
def create_feature_engineering_pipeline(numerical_features, categorical_features):

    num_transformer = Pipeline([
        ("scaler", MinMaxScaler())
    ])

    cat_transformer = Pipeline([
        ("encoder", OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, numerical_features),
        ("cat", cat_transformer, categorical_features)
    ])

    return preprocessor


# Extract feature names after transformation
def get_feature_names(preprocessor, numerical_features, categorical_features):
    # Get names of numerical features (unchanged)
    num_features_out = numerical_features

    # Get names of categorical features after one-hot encoding
    cat_pipeline = preprocessor.named_transformers_["cat"]
    cat_encoder = cat_pipeline.named_steps["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)

    # Combine all feature names
    return np.concatenate([num_features_out, cat_feature_names])


# Experimentation for traditional models with different optimizers
def run_experiment(df, test_size, target, numerical_features, categorical_features, model, save_folder):
    X_train, X_test, y_train, y_test = load_data(df, test_size, target)
    preprocessor = create_feature_engineering_pipeline(numerical_features, categorical_features)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rsquared = r2_score(y_test, y_pred)

    feature_names = get_feature_names(preprocessor, numerical_features, categorical_features)

    # Transform X_test and create DataFrame with correct column names
    X_test_transformed = preprocessor.transform(X_test)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
    X_test_df["y_test"] = y_test.values
    X_test_df["y_pred"] = y_pred

    model_name = model.__class__.__name__
    X_test_df.to_csv(save_folder + model_name + '_baseline.csv')

    return rsquared, pipeline


# # Run experiments
# def main():
#     # Traditional Models with optimizer selection
#     model_configs = [
#         {"model": Ridge(), "params": {"model__alpha": [0.1, 1, 10]}},
#         {"model": SVR(), "params": {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]}},
#         {"model": SGDRegressor(), "params": {"model__alpha": [0.0001, 0.001, 0.01], "model__penalty": ["l2", "l1"]}},
#         {"model": RandomForestRegressor(), "params": {"model__n_estimators": [10, 50, 100]}}
#     ]

#     for config in model_configs:
#         best_params, mse = run_experiment(config["model"], config["params"])
#         print(f"Best Params: {best_params}, MSE: {mse}")

