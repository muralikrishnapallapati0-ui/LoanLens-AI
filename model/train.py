import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'archive', 'train_u6lujuX_CVtuZ9i.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')


def load_and_preprocess(path):
    df = pd.read_csv(path)
    df.drop(columns=['Loan_ID'], inplace=True, errors='ignore')

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())

    if df['Loan_Status'].dtype == object:
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']
    return X, y


def train_models(X_train, y_train, X_test, y_test):
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {'model': model, 'train_acc': train_acc, 'test_acc': test_acc}
        print(f"{name}: Train={train_acc:.4f}, Test={test_acc:.4f}")

    return results


def save_best_model(results):
    best_name = max(results, key=lambda k: results[k]['test_acc'])
    best_model = results[best_name]['model']
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))
    print(f"\nBest model: {best_name} saved.")
    return best_name


if __name__ == '__main__':
    X, y = load_and_preprocess(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = train_models(X_train, y_train, X_test, y_test)
    save_best_model(results)
