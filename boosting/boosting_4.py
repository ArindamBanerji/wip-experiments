import xgboost as xgb

start = time.time()

xgb = xgb.XGBRegressor(max_depth=3, n_estimators=100, 
                       objectvie='reg:squarederror', 
                       booster='gbtree', n_jobs=-1,
                       random_state=0, learning_rate=0.1)

xgb_scores = cross_val_score(xgb, X, y,
                              scoring="neg_mean_squared_error",
                              cv=5, n_jobs=1)

xgb_rmse = np.sqrt(-xgb_scores)
print("Cross-validated RMSE for XGBoost: ", np.mean(xgb_rmse))

end = time.time()
diff = end - start
print('Execution time for XGBoost (in Seconds):', diff)