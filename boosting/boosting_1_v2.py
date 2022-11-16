# Simple compriso of xgboost, lightgbm & catboost

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm
import time

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
file = config['files']['california_housing']

# extract local file name
fnm, exists = get_full_datapath_nm (config['current_proj_dir'], 
                                        config['data_dir_nm'], file)   
print ("full_path nm -from read_df", fnm)
if (exists ==  False) :
    print ("file does not exist", file)        

# load data
df = pd.read_csv(fnm)
df.head()

# set the x, y parms 
X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']

# split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=1)

#CatBoost - original

start = time.time()

cb = CatBoostRegressor(n_estimators=100,
                       loss_function='RMSE',
                       learning_rate=0.1,
                       depth=3, task_type='CPU',
                       random_state=1,
                       verbose=False)

cb.fit(X_train, y_train)
y_pred = cb.predict(X_test)

cb_rmse = np.sqrt(mse(y_test, y_pred))
print("RMSE for CatBoost: ", np.mean(cb_rmse))

end = time.time()

diff = end - start
print('Execution time for CatBoost (in Seconds):', diff)

#XGBoost

start = time.time()

xgbr = xgb.XGBRegressor(max_depth=3, n_estimators=100, 
                           eval_metric=mse, 
                           booster='gbtree', n_jobs=-1,
                           random_state=0, learning_rate=0.1)

xgbr.fit(X_train, y_train)
y_pred = xgbr.predict(X_test)

xgbr_rmse = np.sqrt(mse(y_test, y_pred))
print("RMSE for XGBoost: ", np.mean(xgbr_rmse))

end = time.time()
diff = end - start
print('Execution time for XGBoost (in Seconds):', diff)

#LightGBM

start = time.time()

lgbm = lightgbm.LGBMRegressor(boosting_type='gbdt',
                              max_depth=3,
                              n_estimators=100,
                              learning_rate=0.1,
                              objective='mse',
                              n_jobs=-1, random_state=0)


# lgbm.fit(X_train, y_train)
# y_pred = lgbm.predict(X_test)

lgbm_rmse = np.sqrt(mse(y_test, y_pred))
print()
print("RMSE for LightGBM: ", np.mean(lgbm_rmse))

end = time.time()
diff = end - start
print('Execution time for LightGBM (in Seconds):', diff)

lgbm.fit(X, y)
lightgbm.plot_importance(lgbm)


# Try with  cross validation for each one of the above - first up lightgbm
start = time.time()

lgbm = lightgbm.LGBMRegressor(boosting_type='gbdt',
                              max_depth=3,
                              n_estimators=100,
                              learning_rate=0.1,
                              objective='mse',
                              n_jobs=-1, random_state=0)

lgbm_scores = cross_val_score(lgbm, X, y,
                              scoring="neg_mean_squared_error",
                              cv=5, n_jobs=1)

lgbm_rmse = np.sqrt(-lgbm_scores)
print("Cross-validated RMSE for LightGBM: ", np.mean(lgbm_rmse))

end = time.time()
diff = end - start
print('Execution time for LightGBM with CV (in Seconds):', diff)

# XGBoost - with cv
start = time.time()

xgbr = xgb.XGBRegressor(max_depth=3, n_estimators=100, 
                           eval_metric=mse, 
                           booster='gbtree', n_jobs=-1,
                           random_state=0, learning_rate=0.1)

xgbr_scores = cross_val_score(xgbr, X, y,
                              scoring="neg_mean_squared_error",
                              cv=5, n_jobs=1)

xgbr_rmse = np.sqrt(-xgbr_scores)
print("Cross-validated RMSE for XGBoost: ", np.mean(xgbr_rmse))

end = time.time()
diff = end - start
print('Execution time for XGBoost with CV (in Seconds):', diff)

# catboost cv 
start = time.time()

cb = CatBoostRegressor(n_estimators=100,
                       loss_function='RMSE',
                       learning_rate=0.1,
                       depth=3, task_type='CPU',
                       random_state=1,
                       verbose=False)

cb_scores = cross_val_score(cb, X, y,
                              scoring="neg_mean_squared_error",
                              cv=5, n_jobs=1)

cb_rmse = np.sqrt(-cb_scores)
print("Cross-validated RMSE for CatBoost: ", np.mean(cb_rmse))

end = time.time()
diff = end - start
print('Execution time for CatBoost with CV (in Seconds):', diff)
