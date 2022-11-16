# Import libraries with standard conventions
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

df = pd.read_csv("heart_disease.csv") # Loading the data

X = df.iloc[:,:-1] # Feature matrix in pd.DataFrame format
y = df.iloc[:,-1] # Target vector in pd.Series format

# Shuffling the data
X_shuffle, y_shuffle = shuffle(X, y, random_state=42)

# Instantiate an XGBoost object with hyperparameters
xgb_clf = xgb.XGBClassifier(max_depth=3, n_estimators=100,
                            objective='binary:logistic', booster='gbtree',
                            n_jobs=2, random_state=1)

# Cross-validation with 10 folds
acc_scores = cross_val_score(xgb_clf, X_shuffle, y_shuffle,
                         scoring="accuracy",
                         cv=10, n_jobs=-1)

print("Accuracy Values: ", np.round(acc_scores, 2))
print("Average Accuracy: ", np.mean(acc_scores))