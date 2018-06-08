import pandas as pd
import numpy as np
import warnings
import data
import randomForest
import gradientBoosting
import ridge
import linearRegression
import knn
import MLPRegressor
import svr
import lasso
import lightgbm2
import xgb
warnings.filterwarnings('ignore')

X_train, y_train, X_test, id = data.load_data()
X_t, X_v, y_t, y_v = data.data_val()
print(y_t.size)
print(y_v.size)
r1 = randomForest.predict(X_t, y_t, X_v)
print(1)
r2 = gradientBoosting.predict(X_t, y_t, X_v)
print(1)
r3 = lightgbm2.predict(X_t, y_t, X_v)
print(1)
r4 = lasso.predict(X_t, y_t, X_v)
print(1)
r5 = xgb.predict(X_t, y_t, X_v)
print(1)
r6 = ridge.predict(X_t, y_t, X_v)
print(1)
#r7 = svr.predict(X_t, y_t, X_v)
print(1)

t1 = randomForest.predict(X_t, y_t, X_test)
t2 = gradientBoosting.predict(X_t, y_t, X_test)
t3 = lightgbm2.predict(X_t, y_t, X_test)
t4 = lasso.predict(X_t, y_t, X_test)
t5 = xgb.predict(X_t, y_t, X_test)
t6 = ridge.predict(X_t, y_t, X_test)
#t7 = svr.predict(X_train, y_train, X_train)

X_t=np.concatenate((r1,r2,r3,r4,r5,r6), axis=1)
X_r=np.concatenate((t1,t2,t3,t4,t5,t6), axis=1)

from sklearn.linear_model import Ridge
clf = Ridge(alpha=0)
# from sklearn.ensemble import GradientBoostingRegressor
# clf = GradientBoostingRegressor(n_estimators=10000,
#                                     max_features='sqrt',
#                                     max_depth=4,
#                                     random_state=1,
#                                     learning_rate=0.015,
#                                     min_samples_split=2,
#                                     min_samples_leaf=1,
#                                     subsample=0.2)
clf.fit(X_t, y_v)
result = np.expm1(clf.predict(X_r).reshape(-1,1))

ans = np.hstack((id,result))
ans = pd.DataFrame(ans, columns=['Id','SalePrice'])
ans['Id'] = ans['Id'].astype('Int32')
ans.to_csv('submission.csv',index=False)