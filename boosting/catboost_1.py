from catboost import CatBoostRegressor, Pool

cb = CatBoostRegressor(n_estimators=200,
                       loss_function='RMSE',
                       learning_rate=0.4,
                       depth=3, task_type='CPU',
                       random_state=1,
                       verbose=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=1)

pool_train = Pool(X_train, y_train,
                  cat_features = ['cut', 'color', 'clarity'])

pool_test = Pool(X_test, cat_features = ['cut', 'color', 'clarity'])

cb.fit(pool_train)
y_pred = cb.predict(pool_test)

import numpy as np
from sklearn.metrics import mean_squared_error as mse

cb_rmse = np.sqrt(mse(y_test, y_pred))
print("RMSE in y units:", np.mean(cb_rmse))