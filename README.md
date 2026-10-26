# LoanLens🏦

ML-Loan Eligibility Prediction System web app built with Flask and XGBoost.

## Models Used
| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Decision Tree | - | - |
| Random Forest | - | - |
| KNN | - | - |
| **XGBoost** | **94.7%** | **81.1%** |

## Project Structure
```
SmartLender/
├── app/
│   ├── __init__.py        # Flask app factory
│   ├── routes.py          # Routes & prediction logic
│   ├── templates/
│   │   ├── index.html     # Applicant input form
│   │   └── result.html    # Prediction result page
│   └── static/css/
│       └── style.css      # Stylesheet
├── model/
│   ├── train.py           # Train all 4 models & save best
│   └── best_model.pkl     # Saved XGBoost model (generated)
├── data/
│   └── loan_data.csv      # Dataset (add your own)
├── notebooks/
│   └── eda_and_training.ipynb
├── run.py                 # App entry point
├── Procfile               # IBM Cloud deployment
├── manifest.yml           # IBM Cloud config
└── requirements.txt
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Dataset is already at data/archive/train_u6lujuX_CVtuZ9i.csv

# 3. Train the model
python model/train.py

# 4. Run the Flask app
python run.py
```

Visit `http://localhost:5000`

## Deploy to IBM Cloud

```bash
ibmcloud cf push
```

## Dataset Columns Expected
`Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status`
