import os

class Config:
    basedir = os.path.abspath(os.path.dirname(__file__))
    DB_SQL_PATH = 'sqlite:///' + os.path.join(basedir, 'persistent/service_1.db')
    DB_MONGO_PATH = "mongodb+srv://sachinjoseph054:kOpRxNfjcc1GC74w@asdcluster.id1l6xq.mongodb.net/ASD"
    EYE_COLLECTION ="EyeFeatures"

    # model related(AWS S3)
    EYE_MODEL_FILE ="ASD_model.keras"
    BUCKET_NAME = "asdinsightmodels"
    METADATA_COLLECTION = "ModelMetadata"