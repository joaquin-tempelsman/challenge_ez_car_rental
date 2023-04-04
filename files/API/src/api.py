import sys
sys.path.insert(1,'../src')
from train_api import retrain_api
import os
from pathlib import Path
from lightgbm import LGBMRegressor
import joblib
import pandas as pd
import json
import datetime


def train(trials):
    
    train_api(trials)


#def predict(car_parking_address_city:str, trip_duration:float, date:str, trip_start_at_local_time:str):
def predict(city:str, duration:float, date:str, time:str):

    # expected format example
    # ['Chicago', 210.2, '2017-10-01', '20:20']

    month = datetime.datetime.strptime(date, '%Y-%m-%d').month
    time = datetime.datetime.strptime(time, '%H:%M').hour
    series = pd.Series({'car_parking_address_city': str(city),
                      'trip_duration': float(duration),
                      'month': str(month),
                      'time': str(time) 
                      })   



    data = pd.DataFrame()
    data = data.append(series,ignore_index=True)
    data['month'] = data['month'].astype('category')
    data['time'] = data['time'].astype('category')


    # newest folder in trained models
    BASE_DIR = sorted(Path('trained_models_api/').iterdir(), key=os.path.getmtime)[0]

    # pipeline and model path
    model_file = Path(BASE_DIR).joinpath("model.joblib") 
    pipeline = Path(BASE_DIR).joinpath("pipeline.joblib")

    # load model and pipeline
    pipeline = joblib.load(pipeline)
    model = joblib.load(model_file)

    # transform data
    X_inference = pipeline.transform(data)

    # make inference
    prediction = model.predict(X_inference)


    return prediction