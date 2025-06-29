# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import os, joblib,  missingno


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures

from sklearn.pipeline import Pipeline, FeatureUnion
#from sklearn_features.transformers import DataFrameSelector


from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.neighbors import KNeighborsRegressor


File_PATH = os.path.join(os.getcwd(), 'housing.csv')
df_housing = pd.read_csv(File_PATH)


# Rename (<1H OCEAN)
df_housing['ocean_proximity'] = df_housing['ocean_proximity'].replace('<1H OCEAN', '1H OCEAN')
df_housing['ocean_proximity'].unique()

# Adding new Features 
df_housing['rooms_per_household'] = df_housing['total_rooms'] / df_housing['households']
df_housing['bedrooms_per_rooms'] = df_housing['total_bedrooms'] / df_housing['total_rooms']
df_housing['population_per_household'] = df_housing['population'] / df_housing['households']


X = df_housing.drop(columns='median_house_value', axis=1) # Features
y = df_housing['median_house_value'] # target



X_train, X_test, y_train, y_test=train_test_split(X,y, shuffle=True, test_size=0.2, random_state=42)

num_cols = [col for col in X_train.columns if X_train[col].dtype in ['float32','float64','int32','int64']]
cat_cols = [col for col in X_train.columns if X_train[col].dtype not in ['float32','float64','int32','int64']]

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]
    


num_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(num_cols)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scalar', StandardScaler())
     ])

categ_pipeline = Pipeline(steps=[
     ('selector', DataFrameSelector(cat_cols)),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe',OneHotEncoder())
])


total_pipeline = FeatureUnion(transformer_list=[
    ('num',num_pipeline ),
    ('categ',categ_pipeline)
])
X_train_final = total_pipeline.fit_transform(X_train) #train




def preprocess(X_new):
    '''
    The aim of this function is to process the new instances 
    Returns
    Preprocessed Features ready to make inference by the Model
    '''
    return total_pipeline.transform(X_new)