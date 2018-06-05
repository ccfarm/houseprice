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
warnings.filterwarnings('ignore')

X_train, y_train, X_test, id = data.load_data()
r1 = randomForest.predict(X_train, y_train, X_test)
print(1)
r2 = gradientBoosting.predict(X_train, y_train, X_test)
print(1)
r3 = ridge.predict(X_train, y_train, X_test)
print(1)
r4 = lasso.predict(X_train, y_train, X_test)
print(1)
r5 = knn.predict(X_train, y_train, X_test)
print(1)
r6 = MLPRegressor.predict(X_train, y_train, X_test)
print(1)
#r7 = svr.predict(X_train, y_train, X_test)
print(1)

t1 = randomForest.predict(X_train, y_train, X_train)
t2 = gradientBoosting.predict(X_train, y_train, X_train)
t3 = ridge.predict(X_train, y_train, X_train)
t4 = lasso.predict(X_train, y_train, X_train)
t5 = knn.predict(X_train, y_train, X_train)
t6 = MLPRegressor.predict(X_train, y_train, X_train)
#t7 = svr.predict(X_train, y_train, X_train)

X_t=np.concatenate((t1,t2,t3,t5,t6), axis=1)
X_r=np.concatenate((r1,r2,r3,r5,r6), axis=1)

# from sklearn.linear_model import Ridge
# clf = Ridge(alpha=0.1)
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor(n_estimators=10000,
                                    max_features='sqrt',
                                    max_depth=4,
                                    random_state=1,
                                    learning_rate=0.015,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    subsample=0.2)
clf.fit(X_t, y_train)
result = np.expm1(clf.predict(X_r).reshape(-1,1))

ans = np.hstack((id,result))
ans = pd.DataFrame(ans, columns=['Id','SalePrice'])
ans['Id'] = ans['Id'].astype('Int32')
ans.to_csv('submission.csv',index=False)