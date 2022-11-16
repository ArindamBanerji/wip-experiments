# Import libraries with standard conventions
import numpy as np
import pandas as pd
import xgboost as xgb

df = pd.read_csv("heart_disease.csv") # Loading the data

X = df.iloc[:,:-1] # Feature matrix in pd.DataFrame format
y = df.iloc[:,-1] # Target vector in pd.Series format

# Creating the DMatrix
d_matrix = xgb.DMatrix(data=X, label=y)

# Define parameters
params = {"max_depth":3, "n_estimators":100, "objective":"binary:logistic",
          "booster":"gbtree", "n_jobs":2, "random_state":1}

# Cross-validation with 10 folds
cv = xgb.cv(params, d_matrix, nfold=10, num_boost_round=10, 
            as_pandas=True, shuffle=True, seed=42, metrics="error")

print(cv)
print("---------------------------------------------------------------------")
print("Accuracy Values: ", np.array((1 - cv['test-error-mean'])).round(2))
print("Average Accuracy: ", (1 - cv['test-error-mean']).mean())