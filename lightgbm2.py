import lightgbm as lgb
import data
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def predict(X_train, y_train, X_test):

    clf = lgb.LGBMRegressor(objective='regression',num_leaves=5
                            ,max_depth=4,min_child_weight=1.5,
                            learning_rate=0.01, n_estimators=7200,max_bin = 55,
				bagging_fraction=0.8,bagging_freq=5,feature_fraction=0.2319,feature_fraction_seed=9,bagging_seed=9,min_data_in_leaf=6,min_sum_hessian_in_leaf=11)
    clf.fit(X_train,y_train)
    result = clf.predict(X_test).reshape(-1,1)
    return result

if __name__ == "__main__":
    X_train, y_train, X_test, id = data.load_data()
    result = np.expm1(predict(X_train, y_train, X_test))

    ans = np.hstack((id,result))
    ans = pd.DataFrame(ans, columns=['Id','SalePrice'])
    ans['Id'] = ans['Id'].astype('Int32')
    ans.to_csv('submission.csv',index=False)
