from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

class EyePredictor:
    def __init__(self):
        self.model_path = 'api_server/service/prediction/autism_efficient_net20.h5'  # Specify your model path here
        self.model = self.load_model()

    def load_model(self):
        # Load your EfficientNet model
        model = load_model(self.model_path)
        return model

    def preprocess_image(self, img_path):
        target_size = (224, 224)  # Adjust according to your model's input size
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
        img_array /= 255.0  # Normalize to [0,1]
        return img_array

    def predict(self, img_path):
        try:
            # Preprocess the image
            img_array = self.preprocess_image(img_path)

            # Make prediction
            prediction = self.model.predict(img_array)[0][0]  # Assuming binary classification

            # Interpret prediction
            if prediction >= 0.5:
                result = {'prediction': 'Autistic', 'confidence': float(prediction)}
            else:
                result = {'prediction': 'Not Autistic', 'confidence': float(prediction)}

            return result, 200

        except Exception as e:
            error_message = f'Error predicting image: {str(e)}'
            return {'error': error_message}, 500
