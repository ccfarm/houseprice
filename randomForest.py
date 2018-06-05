import data
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def predict(X_train, y_train, X_test):
    clf = RandomForestRegressor(n_estimators=10000,max_features='sqrt', max_depth=19)
    clf.fit(X_train,y_train)
    result = clf.predict(X_test).reshape(-1,1)
    return result