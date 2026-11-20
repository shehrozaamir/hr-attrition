from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles

ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and threshold saved from notebook
    model_path = 'credit_risk_model.pkl' if os.path.exists('credit_risk_model.pkl') else 'hr_attrition_model.pkl'
    ml_model['model'] = joblib.load(model_path)
    ml_model['threshold'] = joblib.load('best_threshold.pkl')
    yield
    ml_model.clear()

app = FastAPI(lifespan=lifespan)

class EmployeeApplication(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: float
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: float
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: float
    YearsInCurrentRole: int
    YearsSinceLastPromotion: float
    YearsWithCurrManager: int

@app.post('/predict')
def predict(data: EmployeeApplication):
    input_df = pd.DataFrame([data.model_dump()])

    # Log1p transform on skewed columns (as done during training)
    skewed_cols = ['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion']
    for c in skewed_cols:
        input_df[c] = np.log1p(input_df[c])

    probability = float(ml_model['model'].predict_proba(input_df)[:, 1][0])
    prediction = int(probability >= ml_model["threshold"])

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": float(ml_model["threshold"]),
        "Result": "High Risk (Likely to Leave)" if prediction == 1 else "Low Risk (Likely to Stay)"
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
