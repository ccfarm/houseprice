from sklearn.neural_network import MLPRegressor
import data
import numpy as np
import pandas as pd

def predict(X_train, y_train, X_test):
    clf = MLPRegressor(hidden_layer_sizes=(10,10,10,10,10,10,10,10), alpha=20)
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
