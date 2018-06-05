from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import data

def predict(X_train, y_train, X_test):
    clf = GradientBoostingRegressor(n_estimators=10000,
                                    max_features='sqrt',
                                    max_depth=4,
                                    random_state=1,
                                    learning_rate=0.015,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    subsample=0.2)
    clf.fit(X_train,y_train)
    result = clf.predict(X_test).reshape(-1,1)
    return result

if __name__ == "__main__":
    X_train, y_train, X_test, id = data.load_data()
    result = predict(X_train, y_train, X_test)

    ans = np.hstack((id,result))
    ans = pd.DataFrame(ans, columns=['Id','SalePrice'])
    ans['Id'] = ans['Id'].astype('Int32')
    ans.to_csv('submission.csv',index=False)