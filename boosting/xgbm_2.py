# Import libraries with standard conventions
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split

from helper_fe_v2 import (
            get_full_datapath_nm,
            read_df_from_file,
            check_module_members,
            gen_correlation,
            do_bkwd_fwd_selection,
            yaml_path,
            read_yaml_conf,
            remove_duplicates, 
            drop_const_features,
            drop_quasi_const_features ,
            run_randomForestClassifier,
            run_logistic,
            run_randomForestRegressor
)


# Read in the yaml file
config = read_yaml_conf(yaml_path())
file = config['files']['heart_disease']

# extract local file name
fnm, exists = get_full_datapath_nm (config['current_proj_dir'], 
                                        config['data_dir_nm'], file)   
print ("full_path nm -from read_df", fnm)
if (exists ==  False) :
    print ("file does not exist", file)        


df = pd.read_csv(fnm) # Loading the data

X = df.iloc[:,:-1] # Feature matrix in pd.DataFrame format
y = df.iloc[:,-1] # Target vector in pd.Series format

# Making train and test sets for both X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, shuffle=True)

# Creating DMatrices
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

# Parameter dictionary
params = {'max_depth':4, 'objective':'binary:logistic',
          'n_estimators':100, 'booster':'gbtree'} 

# Train the model with train data sets
xgb_clf = xgb.train(params=params, dtrain=dtrain)

preds = xgb_clf.predict(dtest) # Predictions returns as probabilities
y_pred = [round(value) for value in preds]
y_pred = np.array(y_pred).astype(int) # Predictions returns as classes
y_true = y_test # True values

print("Accuracy: ", np.round(accuracy_score(y_true, y_pred), 3))