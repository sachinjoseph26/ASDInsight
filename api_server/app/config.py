import os

class Config:
    # Database
    MONGO_URI = os.getenv('MONGO_URI')
    MONGO_DATABASE_NAME = os.getenv('MONGO_DATABASE_NAME')
    QCHAT_COLLECTION = os.getenv('QCHAT_COLLECTION')
    EYE_COLLECTION = os.getenv('EYE_COLLECTION')
    METADATA_COLLECTION = os.getenv('METADATA_COLLECTION')

    # Model related (AWS S3)
    EYE_MODEL_FILE = os.getenv('EYE_MODEL_FILE')
    AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')
    BUCKET_NAME = os.getenv('BUCKET_NAME')


    # Azure
    AML_WORKSPACE_NAME = os.getenv('AML_WORKSPACE_NAME')
    AZURE_SUBSCRIPTION_ID = os.getenv('AZURE_SUBSCRIPTION_ID')
    AZURE_RESOURCE_GROUP = os.getenv('AZURE_RESOURCE_GROUP')
    MODEL_NAME = os.getenv('MODEL_NAME')
    SERVICE_NAME = os.getenv('SERVICE_NAME')