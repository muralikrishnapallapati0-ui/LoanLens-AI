import os
import joblib
import numpy as np
from flask import Blueprint, render_template, request

main = Blueprint('main', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_model.pkl')

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

GENDER_MAP = {'Male': 1, 'Female': 0}
MARRIED_MAP = {'Yes': 1, 'No': 0}
EDUCATION_MAP = {'Graduate': 0, 'Not Graduate': 1}
SELF_EMPLOYED_MAP = {'Yes': 1, 'No': 0}
PROPERTY_MAP = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
DEPENDENTS_MAP = {'0': 0, '1': 1, '2': 2, '3+': 3}


@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@main.route('/predict', methods=['POST'])
def predict():
    model = load_model()
    if model is None:
        return render_template('result.html', result='Model not found. Please train the model first.', risk='error')

    try:
        features = [
            GENDER_MAP[request.form['gender']],
            MARRIED_MAP[request.form['married']],
            DEPENDENTS_MAP[request.form['dependents']],
            EDUCATION_MAP[request.form['education']],
            SELF_EMPLOYED_MAP[request.form['self_employed']],
            float(request.form['applicant_income']),
            float(request.form['coapplicant_income']),
            float(request.form['loan_amount']),
            float(request.form['loan_term']),
            int(request.form['credit_history']),
            PROPERTY_MAP[request.form['property_area']],
        ]
        prediction = model.predict([features])[0]
        result = 'Approved' if prediction == 1 else 'Rejected'
        risk = 'low' if prediction == 1 else 'high'
    except Exception as e:
        result = f'Prediction error: {str(e)}'
        risk = 'error'

    return render_template('result.html', result=result, risk=risk)
