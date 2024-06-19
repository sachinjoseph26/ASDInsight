from service.data_service.data_service import DataService
from service.data_processing.data_processing import DataProcessing
from service.model_training.model_training import ModelTraining
import cv2
import os

class EyeTracking:
    def __init__(self, config, data_service, data_processing_service):
        self.data_service = data_service
        self.data_processing_service = data_processing_service

      # Directories for image processing
        self.base_dir = "data_collection/upload/"
        self.categories = ['train', 'test', 'valid']
        self.classes = ['Autistic', 'Non_Autistic']

    def convert_to_heatmap(self, img_path, save_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        heatmap_img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite(save_path, heatmap_img)

    def create_fixmap(self, img_path, save_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, fixmap_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(save_path, fixmap_img)

    def process_eye_images(self):
        for category in self.categories:
            for cls in self.classes:
                input_folder = os.path.join(self.base_dir, category, cls)
                heatmap_output_folder = os.path.join(self.base_dir, f"{category}_heatmap", cls)
                fixmap_output_folder = os.path.join(self.base_dir, f"{category}_fixmap", cls)

                os.makedirs(heatmap_output_folder, exist_ok=True)
                os.makedirs(fixmap_output_folder, exist_ok=True)

                for img_name in os.listdir(input_folder):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(input_folder, img_name)

                        # Heatmap conversion
                        heatmap_save_path = os.path.join(heatmap_output_folder, img_name)
                        self.convert_to_heatmap(img_path, heatmap_save_path)

                        # Fixmap creation
                        fixmap_save_path = os.path.join(fixmap_output_folder, img_name)
                        self.create_fixmap(img_path, fixmap_save_path)

        print("Conversion to heatmap and fixmap images completed.")

    

    def get_eye_data(self):
        # Capturing eye data
        collection_name = 'EyeTrackData'
        query = {}  # Add specific query if needed
        # projection = {'_id': 0, 'image_path': 1, 'point_of_gaze': 1}
        projection = {}
        data = self.data_service.fetch_data(collection_name, query, projection)
        return data
    
    def extract_eye_features(self):
        data_list = []
        for category in self.categories:
            for cls in self.classes:
                heatmap_dir = os.path.join(self.base_dir, f'{category}_heatmap', cls)
                fixmap_dir = os.path.join(self.base_dir, f'{category}_fixmap', cls)
        
            # Iterate over the files in the heatmap and fixmap directories
            for heatmap_filename in os.listdir(heatmap_dir):
                if heatmap_filename.endswith('.png') or heatmap_filename.endswith('.jpg'):
                    fixmap_filename = heatmap_filename
                
                heatmap_path = os.path.join(heatmap_dir, heatmap_filename)
                fixmap_path = os.path.join(fixmap_dir, fixmap_filename)
                # Extract features from heatmap and fixmap images
                features = self.data_processing_service.extract_features(heatmap_path, fixmap_path)

                # Insert data into MongoDB collection
                data = {
                    'image_path': heatmap_path,
                    **features,
                    'label': cls  # Assuming the folder name indicates the label
                }
                data_list.append(data)

        # Insert collected data into MongoDB
        self.data_service.insert_data('EyeTrackData', data_list)
        return f'{len(data_list)} records inserted into MongoDB collection EyeTrackData'

    
