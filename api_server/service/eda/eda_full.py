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
from flask import current_app

class EDAService:
    def __init__(self, image_folder):
        self.image_folder = image_folder

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
                        hash_value = hash_func(image)
                        class_label = os.path.basename(subdir)  # Use folder name as class label
                        data.append((file, str(hash_value), class_label))
                        if verbose:
                            current_app.logger.info(f'Processed {file}: {hash_value}')
                    except Exception as e:
                        current_app.logger.error(f'Error processing {file}: {e}')

        df = pd.DataFrame(data, columns=['filename', 'hash', 'label'])
        return df

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
            ('std_scaler', StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ('selector', self.DataFrameSelector(categorical_features)),
            ('imputer', SimpleImputer(strategy="most_frequent")),
            # Add more steps for categorical preprocessing if needed
        ])

        full_pipeline = ColumnTransformer([
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ])

        return full_pipeline.fit_transform(df)

    def perform_eda(self, df):
        sns.pairplot(df)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url

    def t_test(self, group1, group2):
        t_stat, p_value = ttest_ind(group1, group2)
        return t_stat, p_value

    def wilcoxon_test(self, group1, group2):
        statistic, pval = ranksums(group1, group2)
        return statistic, pval

    def full_eda_pipeline(self):
        current_app.logger.info('Starting the full EDA process')

        # Step 1: Process images
        current_app.logger.info('Processing images')
        df = self.process_images()

        # Step 2: Perform EDA
        current_app.logger.info('Performing EDA')
        plot_url = self.perform_eda(df)

        # Step 3: Preprocess data
        current_app.logger.info('Preprocessing data')
        processed_data = self.preprocess_data(df)

        # Step 4: Hypothesis Testing
        current_app.logger.info('Performing hypothesis testing')
        group1 = df[df['label'] == 'Autistic']['hash'].apply(lambda x: int(x, 16))
        group2 = df[df['label'] == 'Non_Autistic']['hash'].apply(lambda x: int(x, 16))

        t_stat, p_value = self.t_test(group1, group2)
        wilcoxon_stat, wilcoxon_pval = self.wilcoxon_test(group1, group2)

        results = {
            "plot_url": plot_url,
            "processed_data": processed_data.tolist(),
            "t_test": {
                "t_stat": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            },
            "wilcoxon_test": {
                "wilcoxon_stat": wilcoxon_stat,
                "p_value": wilcoxon_pval,
                "significant": wilcoxon_pval < 0.05
            }
        }

        current_app.logger.info('Full EDA process completed')
        return results
