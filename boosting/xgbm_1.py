# Import libraries with standard conventions
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart_disease.csv") # Loading the data

X = df.iloc[:,:-1] # Feature matrix in pd.DataFrame format
y = df.iloc[:,-1] # Target vector in pd.Series format

# Making train and test sets for both X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, shuffle=True)

# Instantiate an XGBoost object with hyperparameters
xgb_clf = xgb.XGBClassifier(max_depth=3, n_estimators=100,
                            objective='binary:logistic', booster='gbtree',
                            n_jobs=2, random_state=1)

# Train the model with train data sets
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test) # Predictions
y_true = y_test # True values

print("Accuracy: ", np.round(accuracy_score(y_true, y_pred), 3))
print("\nROC Curve")
print(plot_roc_curve(xgb_clf, X_test, y_test))