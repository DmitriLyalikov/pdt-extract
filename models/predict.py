import pandas as pd
import pickle
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
warnings.filterwarnings('ignore')

# Load the model from models folder
with open("xgboost-wide-beta-semituned-model.pkl", 'rb') as f:
    xgb = pickle.load(f)


# Load the model from models folder
with open("lightgbm-wide-beta-tuned-model.pkl", 'rb') as f:
    lgbm = pickle.load(f)

df = pd.read_csv('../data/test/test-simple-drops.csv')
X = df.drop('image', axis=1)
images = df['image']

# Create empty columns for predictions in the dataset
df['XGBoost_Prediction'] = ''
df['LightGBM_Prediction'] = ''







df = df.join(images)
# Write the updated DataFrame to a CSV file
df.to_csv('output.csv', index=False)  # Adjust the filename as desired