import json

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

def train_model(X_train, y_train):
    with open('best_params_.json', 'r') as f:
        best_params = json.load(f)
    rfc = RandomForestClassifier(
        max_depth=best_params['max_depth'],
        # n_estimators=best_params['n_estimators'],
        # max_features=best_params['max_features'],
        # min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )
    rfc.fit(X_train, y_train)
    return rfc

def tune_model(X, y):
    rfc = RandomForestClassifier(random_state=42)
    params = {
        'max_depth': [2, 5, 10],
        # 'n_estimators': [100, 200, 300, 400, 500],
        # 'max_features': [10, 20, 30 , 40],
        # 'min_samples_leaf': [1, 2, 4]
    }
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    clf = GridSearchCV(estimator=rfc, param_grid=params, return_train_score=False, cv=kf, scoring='accuracy')
    clf.fit(X, y)
    with open('best_params_.json', 'w') as fp:
        json.dump(clf.best_params_, fp) 

def evaluate_model(model, X_test, y_test, float_precesion=4):
    y_pred = model.predict(X_test)
    pd.DataFrame(data={'predicted_label': y_pred, 'true_label': y_test}).to_csv('predictions.csv', index=False)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    return json.loads(
        json.dumps(metrics), parse_float=lambda x: round(float(x), float_precesion)
    )