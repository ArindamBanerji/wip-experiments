import numpy as np
import pandas as pd


# df = pd.read_csv('cali_housing.csv')
X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']

from sklearn.model_selection import train_test_split
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