import joblib  # inside model trainer, you have the model, you can load it using joblib
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        # loading the model present inside model_trainer
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction
        # print(prediction)
    
    
    # once prediction pipeline is done, you can create the frontend user application