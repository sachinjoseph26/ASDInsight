from service.data_service.data_service import DataService
from service.data_processing.data_processing import DataProcessing
from sklearn.preprocessing import StandardScaler
import os
from io import StringIO
import pandas as pd
import numpy as np

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
        df = pd.read_json(StringIO(data))
        # Handling `{'$numberDouble': 'NaN'}` entries for all columns
        df = df.applymap(lambda x: np.nan if isinstance(x, dict) and '$numberDouble' in x and x['$numberDouble'] == 'NaN' else x)

        # If sibling_yes_no is 0, then number_of_sibling and sibling_with_ASD should be 0
        df.loc[df['siblings_yesno'] == 0, ['siblings_number', 'sibling_withASD']] = 0

        # If sibling_yes_no is 1 and number_of_sibling or sibling_with_ASD is null, impute with mean/mode
        mean_siblings = df[df['siblings_yesno'] == 1]['siblings_number'].mean()
        mode_sibling_asd = df[df['siblings_yesno'] == 1]['sibling_withASD'].mode()[0]

        df.loc[df['siblings_yesno'] == 1, 'siblings_number'] = df.loc[df['siblings_yesno'] == 1, 'siblings_number'].fillna(mean_siblings)
        df.loc[df['siblings_yesno'] == 1, 'sibling_withASD'] = df.loc[df['siblings_yesno'] == 1, 'sibling_withASD'].fillna(mode_sibling_asd)

        # For rows where sibling_yes_no is null, decide on a strategy
        # Here, we assume no siblings if sibling_yes_no is null, but this could vary based on context
        df['siblings_yesno'].fillna(0, inplace=True)
        df.loc[df['siblings_yesno'] == 0, ['siblings_number', 'sibling_withASD']] = 0
        #Fill missing birthweight with median
        median_birthweight = df['birthweight'].median()
        df['birthweight'].fillna(median_birthweight, inplace=True)
        # Fill mother's education with mode
        mode_education = df['mothers_education'].mode()[0]
        df['mothers_education'].fillna(mode_education, inplace=True)
        # binary encoding of gender and target variable group
        df['sex'] = df['sex'].apply(lambda x: 0 if x == 2 else 1)
        df['group'] = df['group'].apply(lambda x: 0 if x == 7 else 1)
        # Select columns to normalize (excluding non-numeric columns)
        numeric_columns = df.select_dtypes(include=['number']).columns
        columns_to_exclude = ['group']  # List of columns to exclude
        numeric_columns = [col for col in numeric_columns if col not in columns_to_exclude]
        # Initialize StandardScaler
        scaler = StandardScaler()
        # Fit and transform the data
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        df_json = df.to_json(orient='records')
        return df_json
    
        
