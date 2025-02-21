import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from ray import tune

# Load data
def load_data():
    df = pd.read_csv("your_data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Engineering Pipeline
def create_feature_engineering_pipeline():
    numerical_features = ["num_col1", "num_col2"]
    categorical_features = ["cat_col1", "cat_col2"]

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, numerical_features),
        ("cat", cat_transformer, categorical_features)
    ])

    return preprocessor


# Define a PyTorch Dataset
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define a simple PyTorch MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Function to select optimizer dynamically
def get_optimizer(name, model, lr):
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

# Training function for deep learning models
def train_model(config, X_train, y_train, X_val, y_val):
    model = MLP(X_train.shape[1])
    optimizer = get_optimizer(config["optimizer"], model, config["lr"])
    criterion = nn.MSELoss()

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    for epoch in range(config["epochs"]):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            val_loss += criterion(y_pred, y_batch).item()

    tune.report(loss=val_loss / len(val_loader))

# Hyperparameter tuning for deep learning, including optimizer selection
def tune_deep_learning(X_train, y_train, X_val, y_val):
    search_space = {
        "optimizer": tune.choice(["Adam", "SGD", "RMSprop"]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": tune.choice([10, 20, 30])
    }

    tuner = tune.run(
        lambda config: train_model(config, X_train, y_train, X_val, y_val),
        config=search_space,
        num_samples=10
    )

    best_config = tuner.get_best_config(metric="loss", mode="min")
    return best_config

# Run experiments
def main():
    # Deep Learning Model with optimizer selection
    X_train, X_test, y_train, y_test = load_data()
    best_config = tune_deep_learning(X_train, y_train, X_test, y_test)
    print(f"Best Deep Learning Config: {best_config}")

if __name__ == "__main__":
    main()
