from flask import Flask,redirect
from flask_sqlalchemy import SQLAlchemy
from flask_pymongo import PyMongo
from pymongo import MongoClient
from app.config import Config
from flask_swagger_ui import get_swaggerui_blueprint
from service.data_service.data_service import DataService
from service.data_processing.data_processing import DataProcessing
from service.model_training.model_training import ModelTraining
from service.eye_tracking.eye_tracking import EyeTracking
from service.qchat_screening.qchat10_screening import QchatScreening
from app.api import api_bp


# Flask Initialization
app = Flask(__name__)

config = {
        # Add other configurations here if necessary
}

app.config.from_object(config)

# Initialize MongoDB
app.config["MONGO_URI"] = "mongodb+srv://sachinjoseph054:kOpRxNfjcc1GC74w@asdcluster.id1l6xq.mongodb.net/ASD"
mongo = PyMongo(app)
client = MongoClient(app.config["MONGO_URI"])


# Automatically redirect to Swagger UI
@app.route('/')
def index():
    return redirect('/swagger', code=302)


def intialize_app(configName='config'):
    
    swaggerui_blueprint = get_swaggerui_blueprint(
        '/swagger',
        '/static/swagger.json',
        config={
            'app_name': "ASDInsight"
        })
    app.register_blueprint(swaggerui_blueprint)



    # Initialize services with configuration
    data_service = DataService(app.config, mongo)
    data_processing_service = DataProcessing(app.config)
    model_training_service = ModelTraining(app.config,mongo)
    eye_tracking_service = EyeTracking(app.config,data_service,data_processing_service)
    qchat_screening_service = QchatScreening(app.config,data_service,data_processing_service)

    # db.init_app(app)
    # Add services to the app context
    app.data_service = data_service
    app.data_processing_service = data_processing_service
    app.model_training_service = model_training_service
    app.eye_tracking_service = eye_tracking_service
    app.qchat_screening_service = qchat_screening_service

    # Register the API Blueprint
    app.register_blueprint(api_bp)

    return app

