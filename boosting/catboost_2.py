import lightgbm

lgbm = lightgbm.LGBMRegressor(boosting_type='gbdt',
                              max_depth=3,
                              n_estimators=200,
                              learning_rate=0.4,
                              objective='mse',
                              n_jobs=-1, random_state=0)

lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

from sklearn.metrics import mean_squared_error as mse

lgbm_rmse = np.sqrt(mse(y_test, y_pred))
print("RMSE", np.mean(lgbm_rmse))