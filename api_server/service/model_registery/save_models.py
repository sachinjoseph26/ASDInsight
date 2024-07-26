import os
import boto3
import pymongo
from botocore.exceptions import NoCredentialsError
from app.config import Config

class SaveModels():
    def __init__(self, model):
        self.model = model
        self.s3 = boto3.client('s3', region_name=Config.AWS_REGION)
        self.bucket_name = Config.BUCKET_NAME
        self.s3_key = 'models/' + Config.EYE_MODEL_FILE
        self.mongodb_connect_str = Config.MONGO_URI
        self.client = pymongo.MongoClient(self.mongodb_connect_str)
        self.db = self.client[Config.MONGO_DATABASE_NAME]
        self.collection = self.db[Config.METADATA_COLLECTION]

    def save_model_to_file(self, filename):
        # Upload the file to S3
        try:
            self.s3.upload_file(Config.EYE_MODEL_FILE, self.bucket_name, self.s3_key)
            model_url = f'https://{self.bucket_name}.s3.amazonaws.com/{self.s3_key}'
            print(f"Model file uploaded to S3 at {model_url}")
        except FileNotFoundError:
            print("The file was not found")
        except NoCredentialsError:
            print("Credentials not available")

        # Store model metadata in MongoDB
        model_metadata = {
            'model_name': 'ASD_EYE_MODEL',
            'model_url': model_url
        }
        self.collection.insert_one(model_metadata)
