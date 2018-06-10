from sklearn.linear_model import Ridge
import data
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

def predict(X_train, y_train, X_test):
    clf = Ridge(alpha=25)
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
