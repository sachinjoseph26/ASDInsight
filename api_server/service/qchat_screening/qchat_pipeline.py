from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import numpy as np

# Define custom transformers
class HandleNanValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.applymap(lambda x: np.nan if isinstance(x, dict) and '$numberDouble' in x and x['$numberDouble'] == 'NaN' else x)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class SiblingsProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Handle NaN entries in a specific format
        X = X.applymap(lambda x: np.nan if isinstance(x, dict) and '$numberDouble' in x and x['$numberDouble'] == 'NaN' else x)

        # Fill NaN in siblings_yesno with 0
        X['siblings_yesno'].fillna(0, inplace=True)
        
        # Handle sibling-related columns when siblings_yesno is 0
        X.loc[X['siblings_yesno'] == 0, ['siblings_number', 'sibling_withASD']] = 0

        # Calculate statistics needed for transformation
        self.mean_siblings = X.loc[X['siblings_yesno'] == 1, 'siblings_number'].mean()
        self.mode_sibling_asd = X.loc[X['siblings_yesno'] == 1, 'sibling_withASD'].mode()[0]
        return self

    def transform(self, X):
        # Handle NaN entries in a specific format
        X = X.applymap(lambda x: np.nan if isinstance(x, dict) and '$numberDouble' in x and x['$numberDouble'] == 'NaN' else x)

        # Fill NaN in siblings_yesno with 0
        X['siblings_yesno'].fillna(0, inplace=True)
        
        # Handle sibling-related columns when siblings_yesno is 0
        X.loc[X['siblings_yesno'] == 0, ['siblings_number', 'sibling_withASD']] = 0

        # Impute missing values based on sibling_yesno condition
        X['siblings_number'] = X.apply(lambda row: self.mean_siblings if pd.isna(row['siblings_number']) and row['siblings_yesno'] == 1 else row['siblings_number'], axis=1)
        X['sibling_withASD'] = X.apply(lambda row: self.mode_sibling_asd if pd.isna(row['sibling_withASD']) and row['siblings_yesno'] == 1 else row['sibling_withASD'], axis=1)

       # Fill any remaining NaNs with defaults if necessary
        X['siblings_number'].fillna(0, inplace=True)
        X['sibling_withASD'].fillna(0, inplace=True)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class FillMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Handle NaN entries in a specific format
        X = X.applymap(lambda x: np.nan if isinstance(x, dict) and '$numberDouble' in x and x['$numberDouble'] == 'NaN' else x)

        self.median_birthweight = X['birthweight'].median()
        self.mode_education = X['mothers_education'].mode()[0]
        return self

    def transform(self, X):
        # Handle NaN entries in a specific format
        X = X.applymap(lambda x: np.nan if isinstance(x, dict) and '$numberDouble' in x and x['$numberDouble'] == 'NaN' else x)

        X['birthweight'].fillna(self.median_birthweight, inplace=True)
        X['mothers_education'].fillna(self.mode_education, inplace=True)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class EncodeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['sex'] = X['sex'].apply(lambda x: 0 if x == 2 else 1)
        X['group'] = X['group'].apply(lambda x: 0 if x == 7 else 1)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features
    
class SelectiveScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_exclude=None):
        self.columns_to_exclude = columns_to_exclude if columns_to_exclude else []
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Select numeric columns excluding specified columns
        numeric_columns = X.select_dtypes(include=['number']).columns
        self.columns_to_scale = [col for col in numeric_columns if col not in self.columns_to_exclude]
        self.scaler.fit(X[self.columns_to_scale])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.columns_to_scale] = self.scaler.transform(X_copy[self.columns_to_scale])
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        return input_features
    
# Define the transformation pipeline
qchat_transformation_pipeline = Pipeline(steps=[
    ('handle_nan', HandleNanValues()),
    ('siblings', SiblingsProcessing()),
    ('fill_missing', FillMissingValues()),
    ('encode', EncodeFeatures()),
    ('scaler', SelectiveScaler(columns_to_exclude=['group']))
]).set_output(transform="pandas")


# Define the preprocessing pipeline
def get_qchat_preprocessing_pipeline():
 return qchat_transformation_pipeline

# Define the full pipeline
# def get_qchat_pipeline():
#     qchat_pipeline = Pipeline(steps=[
#         ('preprocessor', qchat_transformation_pipeline),
#         ('model', LogisticRegression())
#     ]).set_output(transform="pandas")
#     return qchat_pipeline
