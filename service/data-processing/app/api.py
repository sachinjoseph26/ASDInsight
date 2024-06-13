from flask import Blueprint, make_response, jsonify, request
from app import app, db
from app.models import Model1

api_bp = Blueprint('api', __name__)


@api_bp.route('/data-processing/capture-eye-data', methods=['POST'])
def api_1():
    # get data from database
    return {"msg": "This is a sample api"}

@api_bp.route('/data-processing/preprocess-eye-data', methods=['POST'])
def api_2():
    data = request.get_json()
    value = data['key']
    return {"msg": f"This is a sample Post with input: {value}"}

# Health check endpoint
@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})