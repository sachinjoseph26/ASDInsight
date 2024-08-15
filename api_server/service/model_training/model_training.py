from flask import Flask, jsonify
import os
import cv2
import numpy as np
from pymongo import MongoClient
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from app.config import Config
import pandas as pd
from sklearn.linear_model import LogisticRegression
from io import StringIO
import joblib
from tensorflow.keras.preprocessing import image_dataset_from_directory

class ModelTraining:
    def __init__(self, config, qchatservice):
        self.config = config
        self.client = MongoClient(config["MONGO_URI"])
        self.db = self.client[config["MONGO_DATABASE_NAME"]]
        self.collection = self.db[config["EYE_COLLECTION"]]
        self.qchatservice = qchatservice

    def create_model(self, model_name):
        input_shape = (224, 224, 3)
        base_model = None
        
        if model_name == 'EfficientNetB4':
            base_model = tf.keras.applications.EfficientNetB4(include_top=False, weights="imagenet", input_shape=input_shape, pooling='max')

        x = base_model.output
        x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.002)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(rate=0.45, seed=42)(x)
        output = layers.Dense(self.class_count, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def train_and_evaluate_model(self, model, train_data, val_data, test_data, model_name):
        early_stopper = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)
        history = model.fit(train_data, epochs=60, validation_data=val_data, callbacks=[early_stopper])

        test_loss, test_accuracy = model.evaluate(test_data)
        print(f"Test Accuracy for {model_name}:", test_accuracy)

        test_data_array = []
        labels_array = []
        for images, labels in test_data:
            test_data_array.append(images.numpy())
            labels_array.append(labels.numpy())

        X_test = np.concatenate(test_data_array, axis=0)
        y_test = np.concatenate(labels_array, axis=0)

        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = tf.keras.utils.to_categorical(y_pred, num_classes=self.class_count)

        print(classification_report(y_test, y_pred))

        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, cmap='crest', annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.show()

    def load_data(self, data_dir):
        train_data = image_dataset_from_directory(os.path.join(data_dir, "train"), batch_size=32, image_size=(224, 224), label_mode='categorical', shuffle=True, seed=42)
        test_data = image_dataset_from_directory(os.path.join(data_dir, "test"), batch_size=32, image_size=(224, 224), label_mode='categorical', shuffle=False, seed=42)
        val_data = image_dataset_from_directory(os.path.join(data_dir, "val"), batch_size=32, image_size=(224, 224), label_mode='categorical', shuffle=False, seed=42)
        
        self.class_names = train_data.class_names
        self.class_count = len(self.class_names)
        
        return train_data, val_data, test_data

    def train_model(self, data_dir, model_name='EfficientNetB4'):
        train_data, val_data, test_data = self.load_data(data_dir)
        model = self.create_model(model_name)
        self.train_and_evaluate_model(model, train_data, val_data, test_data, model_name)
        model.save(f'{model_name}_model.keras')

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
        joblib.dump(model, 'qchat_model.pkl')