from flask import Blueprint, make_response, jsonify, request,  current_app
from werkzeug.utils import secure_filename
from service.eye_tracking.eye_tracking import EyeTracking
from service.model_training.model_training import ModelTraining
# from service.model_deployment.model_deployment import ModelDeployment
from service.prediction.predict import EyePredictor
from service.model_registery.save_models import SaveModels
from service.qchat_screening.qchat10_screening import QchatScreening
import os
import io

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
    eye_tracking = EyeTracking(current_app.config, current_app.data_service, current_app.data_processing_service)

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
    eye_tracking = EyeTracking(current_app.config, current_app.data_service, current_app.data_processing_service)

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
    eye_tracking = EyeTracking(current_app.config, current_app.data_service, current_app.data_processing_service)

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
    model_training = current_app.model_training_service
    try:
        if usecase_name == 'eye_tracking':
            result = model_training.train_eye_tracking_model()
        elif usecase_name == 'qchat':
            result = model_training.train_qchat_model()
    except Exception as e:
        return jsonify({'error': f'Error in Training the model : {str(e)}'}), 500
    if result:
        # Return success message or processed files
        return jsonify({'message': 'Model trained successfully'}), 200


# Endpoint for predicting based on eye data
@api_bp.route('/predict-eyebased', methods=['POST'])
def predict_eye_based():
    file = request.files['file']
    file_content = file.read()
    img = io.BytesIO(file_content)
    eye_predictor = EyePredictor(logger=current_app.logger)
    # Make prediction using EyePredictor
    prediction = eye_predictor.predict(img)
    current_app.logger.info(f'Prediction result: {prediction}')
    # Return prediction result
    return jsonify({'prediction': prediction})

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

     # Initialize EyeTracking class with appropriate services and configurations
    q_chat = QchatScreening(current_app.config, current_app.data_service, current_app.data_processing_service)

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
    qchat_screening = QchatScreening(current_app.config, current_app.data_service, current_app.data_processing_service)
    data = qchat_screening.get_qchat_data()
    return jsonify({"data": data})

@api_bp.route('/qchat-screening/preprocess-qchatdata', methods=['POST'])
def preprocess_qn():
     # Initialize QchatScreening class with appropriate services and configurations
    qchat_screening = QchatScreening(current_app.config, current_app.data_service, current_app.data_processing_service)

    try:
        status = qchat_screening.preprocess_qchatdata()
    except Exception as e:
        return jsonify({'error': f'Error in preprocessing questionnaire responses: {str(e)}'}), 500

    # Return extracted features
    return jsonify({'message': f'Preprocessing of QCHAT-10 data completed successfully : {status}'}), 200


# Health check endpoint
@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})
