from service.data_service.data_service import DataService
from service.data_processing.data_processing import DataProcessing
from sklearn.preprocessing import StandardScaler
import os
from io import StringIO
import pandas as pd
import numpy as np
import joblib
import json
import traceback
# Import the pipeline from q_chat_pipeline
from .qchat_pipeline import get_qchat_preprocessing_pipeline


class QchatScreening:
    def __init__(self, config, data_service, data_processing_service, logger):
        self.config = config
        self.data_service = data_service
        self.data_processing_service = data_processing_service
        self.logger = logger

      # Directories for image processing
        self.base_dir = "api_server/data_collection/upload/qchat"
        self.file_name = "QCHAT_dataset.csv"

       # Load the pre-trained pipeline
        self.qchat_preprocessing_pipeline = "qchat_preprocessing_pipeline.pkl"
        self.qchat_preprocessing_pipeline_dir = "api_server/service/qchat_screening"

    def collect_responses(self):
        print("inside collect responses")
        file_path = os.path.join(self.base_dir, self.file_name)
        df_qchat = pd.read_csv(file_path)
        # Convert DataFrame to dictionary for MongoDB insertion
        data_dict = df_qchat.to_dict(orient='records')
        print(data_dict)
        # Insert data into MongoDB collection
        collection_name = self.config["QCHAT_COLLECTION"]
        self.data_service.insert_data(collection_name, data_dict)
        return f'{len(data_dict)} records inserted into MongoDB collection QuestionsData'
    
    def get_qchat_data(self):
        # Retrieving qchat responses
        collection_name = self.config["QCHAT_COLLECTION"]
        db_path = self.config["MONGO_URI"]
        self.logger.info(f"collection_name : {collection_name}")
        self.logger.info(f"Db path : {db_path}")
        query = {}  # Add specific query if needed
        # projection = {'_id': 0, 'image_path': 1, 'point_of_gaze': 1}
        projection = {}
        data = self.data_service.fetch_data(collection_name, query, projection)
        return data
    
    def preprocess_qchatdata(self):
        try:
            data = self.get_qchat_data()
            df = pd.read_json(StringIO(data))

            self.logger.info("Starting QCHAT preprocessing")
            self.logger.info(f"Current directory: {os.getcwd()}")
            
            qchat_preprocessing_pipeline_path = os.path.join(self.qchat_preprocessing_pipeline_dir, self.qchat_preprocessing_pipeline)
            if os.path.exists(qchat_preprocessing_pipeline_path):
                # Try to load the existing pipeline
                preprocess_pipeline = joblib.load(qchat_preprocessing_pipeline_path)
                self.logger.info("Loading QCHAT existing preprocessing pipeline.")

            else:
                self.logger.info("Existing QCHAT preprocessing pipeline doesn't exist. Creating new pipeline")
                # If the pipeline doesn't exist, create and fit it
                preprocess_pipeline = get_qchat_preprocessing_pipeline()
                preprocess_pipeline.fit(df)
                joblib.dump(preprocess_pipeline, self.qchat_preprocessing_pipeline_dir)
                self.logger.info("Fitted and saved new QCHAT preprocessing pipeline.")


            # Apply the pipeline to preprocess data
            df_transformed = preprocess_pipeline.transform(df)
            self.logger.info("Finishd QCHAT preprocessing")
       
            #Convert DataFrame back to JSON format
            df_json = df_transformed.to_json(orient='records')
            return df_json
    
        except Exception as e:
            error_message = {
                "error": f"Error in preprocessing questionnaire responses: {str(e)}",
                "traceback": traceback.format_exc()
            }
            return error_message
