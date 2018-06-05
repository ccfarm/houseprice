from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import data

def predict(X_train, y_train, X_test):
    clf = KNeighborsRegressor(n_neighbors=11)
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