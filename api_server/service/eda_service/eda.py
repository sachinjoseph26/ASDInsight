import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import ttest_ind, ranksums
import imagehash
from PIL import Image
import io
import base64
import numpy as np
from IPython.display import display
from sklearn import set_config
from sklearn.utils import estimator_html_repr

class EDAService:
    def __init__(self, image_folder, output_file='image_hashes.csv'):
        self.image_folder = image_folder
        self.output_file = output_file

    def process_images(self, hash_algorithm='phash', verbose=True):
        data = []

        hash_functions = {
            'phash': imagehash.phash
        }

        if hash_algorithm not in hash_functions:
            raise ValueError(f"Unsupported hash algorithm. Choose from {list(hash_functions.keys())}.")

        hash_func = hash_functions[hash_algorithm]

        for subdir, dirs, files in os.walk(self.image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_path = os.path.join(subdir, file)
                    try:
                        image = Image.open(image_path)
                        image_resized = self.resize_image(image)
                        hash_value = hash_func(image_resized)
                        class_label = os.path.basename(subdir)  # Use folder name as class label
                        stats = self.extract_statistical_features(image_resized)
                        data.append((file, str(hash_value), class_label, *stats))
                    except Exception as e:
                        print(f'Error processing {file}: {e}')

        columns = ['filename', 'hash', 'label', 'mean', 'std', 'skewness', 'kurtosis']
        df = pd.DataFrame(data, columns=columns)
        return df

    def save_hashes_to_csv(self, df):
        df.to_csv(self.output_file, index=False)
        print(f'Saved image hashes to {self.output_file}')

    def resize_image(self, image, size=(128, 128)):
        return image.resize(size)

    def extract_statistical_features(self, image):
        image_array = np.array(image.convert('L')).flatten()  # Convert to grayscale and flatten
        mean = np.mean(image_array)
        std = np.std(image_array)
        skewness = pd.Series(image_array).skew()
        kurtosis = pd.Series(image_array).kurtosis()
        return [mean, std, skewness, kurtosis]

    def calculate_hamming_distances(self, df):
        hashes = df['hash'].apply(lambda x: imagehash.hex_to_hash(x))
        n = len(hashes)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = hashes[i] - hashes[j]
                distances.append({
                    'hash1': df['hash'].iloc[i],
                    'hash2': df['hash'].iloc[j],
                    'distance': dist
                })
        distance_df = pd.DataFrame(distances)
        return distance_df

    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[self.attribute_names]

    def preprocess_data(self, df):
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = df.select_dtypes(include=['object']).columns

        numerical_pipeline = Pipeline([
            ('selector', self.DataFrameSelector(numerical_features)),
            ('imputer', SimpleImputer(strategy="median")),
        ])

        categorical_pipeline = Pipeline([
            ('selector', self.DataFrameSelector(categorical_features)),
            ('imputer', SimpleImputer(strategy="most_frequent")),
        ])

        full_pipeline = ColumnTransformer([
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ])

        display("Pipeline structure:")
        display(full_pipeline)

        return full_pipeline,full_pipeline.fit_transform(df)

    def perform_eda(self, df):
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found for EDA.")

        sns.pairplot(df[numeric_columns])
        plt.show()  # Display the plot
        img = io.BytesIO()
        fig = plt.savefig(img, format='png')
        return fig

    def t_test(self, group1, group2):
        t_stat, p_value = ttest_ind(group1, group2)
        return t_stat, p_value

    def wilcoxon_test(self, group1, group2):
        statistic, pval = ranksums(group1, group2)
        return statistic, pval


    def full_eda_pipeline(self):
        try:
            # Step 1: Process images
            df = self.process_images()

            # Save hashes to CSV
            self.save_hashes_to_csv(df)

            # Step 2: Perform EDA
            plot_url = self.perform_eda(df)

            # Step 3: Preprocess data
            pipeline,processed_data = self.preprocess_data(df)#.toarray()

            # Step 4: Perform statistical tests

            df['hash_int'] = df['hash'].apply(lambda x: int(str(x), 16))

            ## Separate the hash values by class
            autistic_hashes = df[df['label'] == 'autistic']['hash_int']
            non_autistic_hashes = df[df['label'] == 'non_autistic']['hash_int']
#
            ## Perform t-test
#
            t_stat, p_value = self.t_test(autistic_hashes, non_autistic_hashes)
#
            ## Perform Wilcoxon rank-sum test
#
            wilcoxon_stat, wilcoxon_pval = self.wilcoxon_test(autistic_hashes, non_autistic_hashes)

            results = {
                "plot_url": plot_url,
                "t_test": {
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                },
                "wilcoxon_test": {
                    "wilcoxon_stat": wilcoxon_stat,
                    "p_value": wilcoxon_pval,
                    "significant": wilcoxon_pval < 0.05
                },
                "processed_data": processed_data.tolist()
            }

            return results

        except Exception as e:
            return {"error": f"Error in EDA: {str(e)}"}