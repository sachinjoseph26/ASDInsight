import numpy as np
import cv2
import tensorflow as tf
from pymongo import MongoClient
from app.config import Config

class EyePredictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(Config.MODEL_PATH)
        self.client = MongoClient(Config.DB_MONGO_PATH)
        self.db = self.client[Config.DB_NAME]
        self.collection = self.db[Config.EYE_COLLECTION]

    def predict(self, image_file):
        # Process the image
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=-1)
        image = image / 255.0

        # Normalize the path for consistency
        image_filename = image_file.filename
        normalized_image_path = 'data_collection/upload/' + image_filename.replace('\\', '/')

        # Fetch the corresponding record from MongoDB
        record = self.collection.find_one({'image_path': normalized_image_path})

        if record is None:
            return {'error': 'Image features not found in database'}, 404

        # Extract features from the record
        features = [
            record['num_fixations'],
            record['fixation_density'],
            record['mean_intensity_heatmap'],
            record['max_intensity_heatmap'],
            record['min_intensity_heatmap'],
            record['mean_intensity_fixmap'],
            record['max_intensity_fixmap'],
            record['min_intensity_fixmap']
        ]

        # Make prediction
        prediction = self.model.predict([np.array([image]), np.array([features])])[0][0]
        result = 'Autistic' if prediction < 0.5 else 'Non-Autistic'
        
        return {'prediction': result}
