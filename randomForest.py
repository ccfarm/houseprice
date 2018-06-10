import data
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def predict(X_train, y_train, X_test):
    clf = RandomForestRegressor(n_estimators=10000,max_features='sqrt', max_depth=19,
                                random_state=1,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                )
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
