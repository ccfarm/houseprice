import pandas as pd
import numpy as np
import warnings
import data
import xgb
import gradientBoosting
import lightgbm2
import randomForest
import ridge
import lasso
import elastic


warnings.filterwarnings('ignore')

X_train, y_train, X_test, id = data.load_data()
batch = 200
i = 0
while batch * i < X_train.shape[0]:
    start = batch * i
    end = min(batch * (i+1), X_train.shape[0])
    print("batch:", start, end)
    X_1 = pd.concat([X_train[:start], X_train[end:]], axis=0, ignore_index=True)
    y_1 = pd.concat([y_train[:start], y_train[end:]], axis=0, ignore_index=True)
    X_2 = X_train[start:end]
    #y_2 = y_train[start:end]

    t1 = xgb.predict(X_1, y_1, X_2)
    t2 = gradientBoosting.predict(X_1, y_1, X_2)
    t3 = lightgbm2.predict(X_1, y_1, X_2)
    t4 = randomForest.predict(X_1, y_1, X_2)
    t5 = ridge.predict(X_1, y_1, X_2)
    t6 = lasso.predict(X_1, y_1, X_2)
    t7 = elastic.predict(X_1, y_1, X_2)

    if i == 0:
        X_transfer = np.concatenate((t1,t2,t3,t4,t5,t6,t7), axis=1)
    else:
        X_temp = np.concatenate((t1,t2,t3,t4,t5,t6,t7), axis=1)
        X_transfer = np.concatenate((X_transfer, X_temp), axis=0)
    i += 1

t1 = xgb.predict(X_train, y_train , X_test)
t2 = gradientBoosting.predict(X_train, y_train , X_test)
t3 = lightgbm2.predict(X_train, y_train , X_test)
t4 = randomForest.predict(X_train, y_train , X_test)
t5 = ridge.predict(X_train, y_train , X_test)
t6 = lasso.predict(X_train, y_train , X_test)
t7 = elastic.predict(X_train, y_train , X_test)

X_test_transfer = np.concatenate((t1,t2,t3,t4,t5,t6,t7), axis=1)

# from sklearn.linear_model import Ridge
# clf = Ridge(alpha=0)

import xgboost as xgb
clf = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)

clf.fit(X_transfer, y_train)
result = np.expm1(clf.predict(X_test_transfer).reshape(-1,1))


ans = np.hstack((id,result))
ans = pd.DataFrame(ans, columns=['Id','SalePrice'])
ans['Id'] = ans['Id'].astype('Int32')
ans.to_csv('submission.csv',index=False)
