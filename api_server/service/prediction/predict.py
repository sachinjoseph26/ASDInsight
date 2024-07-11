from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import get_custom_objects
import io

# Define and register the swish activation
def swish(x):
    return x * tf.nn.sigmoid(x)

get_custom_objects().update({'swish': swish})

# Define and register the FixedDropout layer
class FixedDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

get_custom_objects().update({'FixedDropout': FixedDropout})

class EyePredictor:
    def __init__(self):
        self.model_path = 'autism_efficient_net20.h5'  # Specify your model path here
        self.model = self.load_model()

    def load_model(self):
        # Load your EfficientNet model with custom objects
        model = load_model(self.model_path, custom_objects={'swish': swish, 'FixedDropout': FixedDropout})
        return model

    def preprocess_image(self, img):
        target_size = (224, 224)  # Adjust according to your model's input size
        if isinstance(img, str):
            img = image.load_img(img, target_size=target_size)
        elif isinstance(img, io.BytesIO):
            img = image.load_img(img, target_size=target_size)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
        img_array /= 255.0  # Normalize to [0,1]
        return img_array

    def predict(self, img):
        img_array = self.preprocess_image(img)
        # Make prediction
        prediction = self.model.predict(img_array)[0][0]
        # Interpret prediction
        if prediction >= 0.5:
            result = 'Autistic'
        else:
            result = 'Not Autistic'
        return result
