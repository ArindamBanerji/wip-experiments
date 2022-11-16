# Simple compriso of xgboost, lightgbm & catboost

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

config = read_yaml_conf(yaml_path())
file = config['files']['california_housing']

fnm, exists = get_full_datapath_nm (config['current_proj_dir'], 
                                        config['data_dir_nm'], file)   
print ("full_path nm -from read_df", fnm)

if (exists ==  False) :
    print ("file does not exist", file)        

# load data
df = pd.read_csv(fnm)
df.head()

# df = pd.read_csv('cali_housing.csv')
X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=1)

#CatBoost
from catboost import CatBoostRegressor
import time

start = time.time()

cb = CatBoostRegressor(n_estimators=100,
                       loss_function='RMSE',
                       learning_rate=0.1,
                       depth=3, task_type='CPU',
                       random_state=1,
                       verbose=False)

cb.fit(X_train, y_train)
y_pred = cb.predict(X_test)

from sklearn.metrics import mean_squared_error as mse

cb_rmse = np.sqrt(mse(y_test, y_pred))
print("RMSE for CatBoost: ", np.mean(cb_rmse))

end = time.time()
diff = end - start
print('Execution time for CatBoost (in Seconds):', diff)

#XGBoost
import xgboost as xgb

start = time.time()

xgb = xgb.XGBRegressor(max_depth=3, n_estimators=100, 
                           objectvie='reg:squarederror', 
                           booster='gbtree', n_jobs=-1,
                           random_state=0, learning_rate=0.1)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

xgb_rmse = np.sqrt(mse(y_test, y_pred))
print("RMSE for XGBoost: ", np.mean(xgb_rmse))

end = time.time()
diff = end - start
print('Execution time for XGBoost (in Seconds):', diff)

#LightGBM
import lightgbm

start = time.time()

lgbm = lightgbm.LGBMRegressor(boosting_type='gbdt',
                              max_depth=3,
                              n_estimators=100,
                              learning_rate=0.1,
                              objective='mse',
                              n_jobs=-1, random_state=0)


lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

lgbm_rmse = np.sqrt(mse(y_test, y_pred))
print()
print("RMSE for LightGBM: ", np.mean(lgbm_rmse))

end = time.time()
diff = end - start
print('Execution time for LightGBM (in Seconds):', diff)