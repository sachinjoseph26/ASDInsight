from flask import Flask, jsonify
import os
import cv2
import numpy as np
from pymongo import MongoClient
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from app.config import Config
import pandas as pd
from sklearn.linear_model import LogisticRegression
from io import StringIO
import joblib


class ModelTraining:

    def __init__(self, config,qchatservice,logger):
        self.config = config
        #self.mongo = mongo
        self.logger = logger
        self.client = MongoClient(config["MONGO_URI"])
        self.db = self.client[config["MONGO_DATABASE_NAME"]]
        self.collection = self.db[config["EYE_COLLECTION"]]
        self.qchatservice = qchatservice
    
    # Load data function
    def load_image_and_features(self, heatmap_path, record):
        # Load heatmap image
        image = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))  # Resize if needed
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = image / 255.0  # Normalize

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
        return image, features
    

    def load_data(self, base_dir, categories, classes):
        data = []
        labels = []
        records = self.collection.find()
        record_dict = {record['image_path'].replace('\\', '/'): record for record in records}

        for category in categories:
            for cls in classes:
                heatmap_dir = os.path.normpath(os.path.join(base_dir, f'{category}_heatmap', cls))
                for heatmap_filename in os.listdir(heatmap_dir):
                    if heatmap_filename.endswith('.png') or heatmap_filename.endswith('.jpg'):
                        heatmap_path = os.path.normpath(os.path.join(heatmap_dir, heatmap_filename)).replace('\\', '/')
                        record = record_dict.get(heatmap_path)
                        if record is not None and len(record) == 11:
                            image, features = self.load_image_and_features(heatmap_path, record)  # Use self.load_image_and_features here
                            data.append((image, features))
                            labels.append(0 if cls == 'Autistic' else 1)

        return data, labels
    
    def split_data(self,data):
        images = np.array([d[0] for d in data])
        features = np.array([d[1] for d in data])
        return images, features

    # Define the model
    def create_cnn_model(self,input_shape):
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape, pooling='avg')
        base_output = base_model.output
        base_output = layers.Dense(64, activation='relu')(base_output)
        base_output = layers.Dense(32, activation='relu')(base_output)
        return Model(inputs=base_model.input, outputs=base_output)

    def create_feature_model(self,input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation='relu')(x)
        return Model(inputs, x)

    def train_eye_tracking_model(self):
        # Load data
        base_dir = 'data_collection/upload/'
        train_data, train_labels = self.load_data(base_dir, ['train'], ['Autistic', 'Non_Autistic'])
        valid_data, valid_labels = self.load_data(base_dir, ['valid'], ['Autistic', 'Non_Autistic'])
        test_data, test_labels = self.load_data(base_dir, ['test'], ['Autistic', 'Non_Autistic'])

        train_images, train_features = self.split_data(train_data)
        valid_images, valid_features = self.split_data(valid_data)
        test_images, test_features = self.split_data(test_data)

        # Convert to numpy arrays
        train_images_np = np.array(train_images)
        train_features_np = np.array(train_features)
        valid_images_np = np.array(valid_images)
        valid_features_np = np.array(valid_features)
        test_images_np = np.array(test_images)
        test_features_np = np.array(test_features)
        train_labels_np = np.array(train_labels)
        valid_labels_np = np.array(valid_labels)
        test_labels_np = np.array(test_labels)

        image_input = layers.Input(shape=(224, 224, 1))
        feature_input = layers.Input(shape=(8,))

        cnn_model = self.create_cnn_model((224, 224, 3))  # Change the input shape to (224, 224, 3) for EfficientNetB0
        image_output = cnn_model(layers.Conv2D(3, (1, 1))(image_input))  # Add Conv2D layer to match input shape

        feature_model = self.create_feature_model((8,))
        feature_output = feature_model(feature_input)

        combined = layers.concatenate([image_output, feature_output])
        x = layers.Dense(64, activation='relu')(combined)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[image_input, feature_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit([train_images_np, train_features_np], train_labels_np, validation_data=([valid_images_np, valid_features_np], valid_labels_np), epochs=10, batch_size=32, callbacks=[early_stopping])

        model.save('ASD_model.keras')
    def get_qchat_features_labels(self, data):
        # Assuming the last column is the label
        features = data.drop(columns=['group','child_id'])
        labels = data['group']
        return features, labels
    
    def train_qchat_model(self):
        preprocessed_data = self.qchatservice.preprocess_qchatdata()
        df_preprocessed = pd.read_json(StringIO(preprocessed_data))
        train_features,train_labels = self.get_qchat_features_labels(df_preprocessed)

        model = LogisticRegression()
        model.fit(train_features, train_labels)
        
        # Save the model
        joblib.dump(self.qchat_pipeline, 'qchat_pipeline.pkl')