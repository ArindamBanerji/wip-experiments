from shapash.explainer.smart_explainer import SmartExplainer

y_pred_as_series = pd.Series(y_pred, index=X_test.index, dtype=np.int)

xpl = SmartExplainer() # Creating xpl object
xpl.compile(x=X_test, model=xgb_clf, y_pred=y_pred_as_series)
app = xpl.run_app() # Launch the app