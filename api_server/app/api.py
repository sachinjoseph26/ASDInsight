from flask import Blueprint, make_response, jsonify, request,  current_app
from werkzeug.utils import secure_filename
from service.eye_tracking.eye_tracking import EyeTracking
from service.model_training.model_training import ModelTraining
# from service.model_deployment.model_deployment import ModelDeployment
from service.prediction.predict import EyePredictor
from service.prediction.QCHATPredictor import QCHATPredictor
from service.model_registery.save_models import SaveModels
from service.qchat_screening.qchat10_screening import QchatScreening
from service.eda_service.eda import EDAService
import os
import io
import json
import numpy as np

api_bp = Blueprint('api', __name__)

# Define base directory and allowed image extensions
base_dir = "data_collection/upload/"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# Define allowed categories and classes
categories = ['train', 'test', 'valid']
classes = ['Autistic', 'Non_Autistic']

@api_bp.route('/data-processing/get-eye-data', methods=['GET'])
def get_eye_data():

     # Initialize EyeTracking class with appropriate services and configurations
    eye_tracking = current_app.eye_tracking_service

    data = eye_tracking.get_eye_data()
    return jsonify({"data": data})
    return {"msg": "This is a sample api"}

@api_bp.route('/eye-tracking/upload-eye-image', methods=['POST'])
def upload_image():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    category = request.form.get('category')
    cls = request.form.get('class')

    # Validate category and class
    if category not in categories:
        return jsonify({'error': f'Invalid category. Allowed categories: {categories}'}), 400

    if cls not in classes:
        return jsonify({'error': f'Invalid class. Allowed classes: {classes}'}), 400

    # Check if the file has an allowed extension
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format'}), 400

    # Create directories if they don't exist
    category_dir = os.path.join(base_dir, category, cls)
    os.makedirs(category_dir, exist_ok=True)

    # Save the uploaded file to the appropriate directory
    filename = secure_filename(file.filename)
    file_path = os.path.join(category_dir, filename)
    file.save(file_path)

    # Return success message or saved file path
    return jsonify({'message': 'Image uploaded successfully',
                    'file_path': file_path}), 200

@api_bp.route('/eye-tracking/process-eye-images', methods=['POST'])
def process_eye_images():
    current_app.logger.info('Received request to process eye images')
 # Check if the upload directory exists
    if not os.path.exists(base_dir):
        current_app.logger.error('Upload directory does not exist')
        return jsonify({'error': 'Upload directory does not exist'}), 500

     # Initialize EyeTracking class with appropriate services and configurations
    eye_tracking = current_app.eye_tracking_service

    # Process the uploaded image
    try:
        eye_tracking.process_eye_images()
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

    # Return success message or processed files
    current_app.logger.info('Image processing completed successfully')
    return jsonify({'message': 'Image processing completed successfully'}), 200

@api_bp.route('/eye-tracking/extract-eye-features', methods=['POST'])
def extract_features():
     # Initialize EyeTracking class with appropriate services and configurations
    eye_tracking = current_app.eye_tracking_service

    try:
        status = eye_tracking.extract_eye_features()
    except Exception as e:
        return jsonify({'error': f'Error extracting features from image: {str(e)}'}), 500

    # Return extracted features
    return jsonify({'message': f'Feature extraction completed successfully : {status}'}), 200

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# model training endpoint
@api_bp.route('/model_training/model-training', methods=['POST'])
def train_model():
    data = request.json
    usecase_name = data.get('usecase_name')
    data_dir = data.get('data_dir')  # Get the data directory from the request

    model_training = current_app.model_training_service
    try:
        if usecase_name == "eye_tracking":
            if not data_dir:
                current_app.logger.error('Data directory not provided for eye_tracking use case')
                return jsonify({'error': 'Data directory not provided for eye_tracking use case'}), 400
            
            current_app.logger.info(f'Training Started for : {usecase_name}')
            result = model_training.train_eye_tracking_model(data_dir=data_dir)
            current_app.logger.info(f'Training Completed for : {usecase_name}')
        
        elif usecase_name == "qchat":
            current_app.logger.info(f'Training Started for : {usecase_name}')
            result = model_training.train_qchat_model()
            current_app.logger.info(f'Training Completed for : {usecase_name}')
        else:
            current_app.logger.error(f'Invalid use case name provided: {usecase_name}')
            return jsonify({'error': 'Invalid use case name provided'}), 400
    except Exception as e:
        return jsonify({'error': f'Error in Training the model: {str(e)}'}), 500

    return jsonify({'message': 'Model trained successfully'}), 200 if result else jsonify({'error': 'Model training failed'}), 500



# Endpoint for predicting based on eye data
@api_bp.route('/predict-eyebased', methods=['POST'])
def predict_eye_based():
    current_app.logger.info(f'Prediction Starting')
    file = request.files['file']
    file_content = file.read()
    img = io.BytesIO(file_content)
    eye_predictor = EyePredictor(logger=current_app.logger)
    # Make prediction using EyePredictor
    prediction = eye_predictor.predict(img)
    current_app.logger.info(f'Prediction result: {prediction}')
    # Return prediction result
    return jsonify({'prediction': prediction})

@api_bp.route('/qchat-screening/predict-qchat-asdrisk', methods=['POST'])
def predict_asd_risk():
    current_app.logger.info(f'QCHAT based Prediction Starting')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.json'):
        try:
            data = json.load(file)
            current_app.logger.info(f'QCHAT based Prediction {data}')
            # Process the JSON data and make predictions
            predictor = QCHATPredictor(current_app.logger)
            prediction = predictor.predict_qchat(data)
            
            return jsonify({'prediction': prediction}), 200
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON file'}), 400
    else:
        return jsonify({'error': 'Invalid file type'}), 400

# Model upload to S3 endpoint
@api_bp.route('/upload_model', methods=['POST'])
def upload_model():
    model_file = request.files['model']
    save_model = SaveModels(model_file)
    save_model.save_model_to_file()
    return jsonify({'message': 'Model uploaded successfully'}), 200
# # Model deployment endpoint
# @api_bp.route('/deploy_model_azure', methods=['POST'])
# def deploy_model():
#     model_s3_path = request.form.get('model_s3_path')
#     model_deployment = ModelDeployment(model_s3_path)
#     scoring_uri = model_deployment.deploy_model()
#     if scoring_uri:
#                 return jsonify({'message': 'Model deployed successfully', 'scoring_uri': scoring_uri}), 200
#     else:
#                 return jsonify({'message': 'Model deployment failed'}), 500
@api_bp.route('/qchat-screening/collect-qchatdata', methods=['POST'])
def collect_qchatdata():
    if not os.path.exists(base_dir):
        return jsonify({'error': 'Upload directory does not exist'}), 500

     # Initialize QChat Screening service
    q_chat = current_app.qchat_screening_service

    # Process the uploaded image
    try:
        q_chat.collect_responses()
    except Exception as e:
        return jsonify({'error': f'Error collecting responses: {str(e)}'}), 500

    # Return success message or processed files
    return jsonify({'message': 'Responses collected successfully'}), 200

@api_bp.route('/qchat-screening/get-qchat-data', methods=['GET'])
def get_qndata():
    # Initialize QchatScreening class with appropriate services and configurations
    qchat_screening = current_app.qchat_screening_service
    data = qchat_screening.get_qchat_data()
    return jsonify({"data": data})

@api_bp.route('/qchat-screening/preprocess-qchatdata', methods=['POST'])
def preprocess_qn():
     # Initialize QchatScreening class with appropriate services and configurations
    qchat_screening = current_app.qchat_screening_service

    try:
        status = qchat_screening.preprocess_qchatdata()
    except Exception as e:
        return jsonify({'error': f'Error in preprocessing questionnaire responses: {str(e)}'}), 500

    # Return extracted features
    return jsonify({'message': f'Preprocessing of QCHAT-10 data completed successfully : {status}'}), 200

# Endpoint for full EDA

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NumpyEncoder, self).default(obj)
        
    
@api_bp.route('/full_eda', methods=['POST'])
def full_eda():
    data = request.json
    if 'image_folder' not in data:
        return jsonify({"error": "No image folder specified"}), 400

    try:
        current_app.logger.info('EDA pipeline started')
        image_folder = data['image_folder']
        eda_service = EDAService(image_folder)
        results = eda_service.full_eda_pipeline()
        current_app.logger.info('EDA pipeline completed successfully')
        return json.dumps(results, cls=NumpyEncoder), 200
    except Exception as e:
        current_app.logger.error(f'Error in EDA pipeline: {str(e)}')
        return jsonify({"error": str(e)}), 500


# Health check endpoint
@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})
