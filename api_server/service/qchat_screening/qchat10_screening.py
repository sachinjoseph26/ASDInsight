from service.data_service.data_service import DataService
from service.data_processing.data_processing import DataProcessing
import os
import pandas as pd

class QchatScreening:
    def __init__(self, config, data_service, data_processing_service):
        self.data_service = data_service
        self.data_processing_service = data_processing_service

      # Directories for image processing
        self.base_dir = "data_collection/upload/qchat"
        self.file_name = "QCHAT_dataset.csv"

    def collect_responses(self):
        print("inside collect responses")
        file_path = os.path.join(self.base_dir, self.file_name)
        df_qchat = pd.read_csv(file_path)
        # Convert DataFrame to dictionary for MongoDB insertion
        data_dict = df_qchat.to_dict(orient='records')
        print(data_dict)
        # Insert data into MongoDB collection
        self.data_service.insert_data('QuestionsData', data_dict)
        return f'{len(data_dict)} records inserted into MongoDB collection QuestionsData'
    
    def get_qchat_data(self):
        # Retrieving qchat responses
        collection_name = 'QuestionsData'
        query = {}  # Add specific query if needed
        # projection = {'_id': 0, 'image_path': 1, 'point_of_gaze': 1}
        projection = {}
        data = self.data_service.fetch_data(collection_name, query, projection)
        return data
    
    def preprocess_qchatdata(self):
        data = self.get_qchat_data()
        print(data)
