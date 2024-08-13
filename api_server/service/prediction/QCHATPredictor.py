import numpy as np
import pandas as pd
import logging
import joblib

class QCHATPredictor:
    def __init__(self,  logger: logging.Logger):
        self.model_path = 'C:\\Users\\hemas\\Documents\\Applied_AI_and_ML_Courses\\Projects_In_Machine_Learning\\ASDInsight\\models\\qchat_LR_model.pkl'  # Specify your model path here
        self.scaler_path = 'C:\\Users\\hemas\\Documents\\Applied_AI_and_ML_Courses\\Projects_In_Machine_Learning\\ASDInsight\\models\\qchat_standard_scaler.pkl'
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.logger = logger
    def normalize_data(self, input_data):
        # Assuming the DataFrame contains all the required features
        # Normalize the data using the pre-fitted StandardScaler
        df = pd.DataFrame([input_data])
        features = df.values
        normalized_features = self.scaler.transform(features)
        return normalized_features
    def predict_qchat(self, input_data):
        self.logger.info("Inside qchat predict method")
        features_array = self.normalize_data(input_data)
        # Make prediction
        prediction = self.model.predict(features_array)
        self.logger.info(f"QCHAT prediction {prediction}")
        Qchat_score = input_data.get("Sum_QCHAT")
        # Assuming binary classification with 0 and 1
        # Interpret prediction
        if prediction >= 0.5:
            result = f'QCHAT Score: {Qchat_score}. There is a high risk the child might suffer from Autism.'
        else:
            result = f'QCHAT Score: {Qchat_score}. The risk is very low. Child is not Autistic!'
        return result
