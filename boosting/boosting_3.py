import numpy as np
import pandas as pd

df = pd.read_csv('cali_housing.csv')
X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']

import lightgbm
import time

start = time.time()

lgbm = lightgbm.LGBMRegressor(boosting_type='gbdt',
                              max_depth=3,
                              n_estimators=100,
                              learning_rate=0.1,
                              objective='mse',
                              n_jobs=-1, random_state=0)

from sklearn.model_selection import cross_val_score

lgbm_scores = cross_val_score(lgbm, X, y,
                              scoring="neg_mean_squared_error",
                              cv=5, n_jobs=1)

lgbm_rmse = np.sqrt(-lgbm_scores)
print("Cross-validated RMSE for LightGBM: ", np.mean(lgbm_rmse))

end = time.time()
diff = end - start
print('Execution time for LightGBM (in Seconds):', diff)

lgbm.fit(X, y)
lightgbm.plot_importance(lgbm)