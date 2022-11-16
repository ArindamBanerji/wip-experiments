import lightgbm

lgbm = lightgbm.LGBMRegressor(boosting_type='gbdt',
                              max_depth=3,
                              n_estimators=200,
                              learning_rate=0.4,
                              objective='mse',
                              n_jobs=-1, random_state=0)

for col in X.select_dtypes(include=['object']):
  X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=1)

lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

from sklearn.metrics import mean_squared_error as mse

lgbm_rmse = np.sqrt(mse(y_test, y_pred))
print("RMSE", np.mean(lgbm_rmse))