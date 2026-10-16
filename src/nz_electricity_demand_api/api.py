from fastapi import FastAPI
import joblib
import pandas as pd
from nz_electricity_demand_api.train import MODEL_PATH
from nz_electricity_demand_api.data import add_features, FEATURES

# Creating FastAPI object
app = FastAPI()
# Load model
model = joblib.load(MODEL_PATH)


@app.get("/predict")
def predict(trading_date: str, trading_period: int):
    # Get features from inputs
    features_df = add_features(pd.DataFrame({"trading_date":[trading_date], "trading_period": [trading_period]}))
    result = model.predict(features_df[FEATURES])
    return {"trading_date": trading_date, "trading_period": trading_period, "demand_mw": float(result[0])}


