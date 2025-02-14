import json

import pandas as pd
from sklearn.model_selection import train_test_split

from metrics_and_plots import plot_confusion_matrix, save_metrics
from model import evaluate_model, train_model, tune_model
from utils import PROCESSED_DATASET, TARGET_COL

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]
    return X, y

def main():
    X, y = load_data(PROCESSED_DATASET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    json.dumps(metrics, indent=2)
    save_metrics(metrics)
    plot_confusion_matrix(model, X_test, y_test)

if __name__ == "__main__":
    main()