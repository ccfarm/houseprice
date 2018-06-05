import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import norm, skew
from scipy.special import boxcox1p

def load_data():
    data_train = pd.read_csv("train.csv")
    data_test = pd.read_csv("test.csv")
    train_shape = data_train.shape
    test_shape = data_test.shape

    data = pd.concat([data_train, data_test], axis=0, ignore_index=True)

    tmp = data.MSSubClass
    X = pd.get_dummies(tmp, prefix='MSSubClass_', dummy_na=True)

    tmp = data.MSZoning
    tmp = pd.get_dummies(tmp, prefix='MSZoning_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.LotFrontage
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.LotArea
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Street
    tmp = tmp.map({'Grvl': 0, 'Pave': 1})
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Alley
    tmp = pd.get_dummies(tmp, prefix='Alley_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.LotShape
    tmp = pd.get_dummies(tmp, prefix='LotShape_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.LandContour
    tmp = pd.get_dummies(tmp, prefix='LandContour_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Utilities
    tmp = tmp.map({'ELO': 0, 'NoSeWa': 1, 'NoSewr':2,'AllPub':3})
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.LotConfig
    tmp = pd.get_dummies(tmp, prefix='LotConfig_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.LandSlope
    tmp = tmp.map({'Sev': 0, 'Mod': 1, 'Gtl':2})
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Neighborhood
    tmp = pd.get_dummies(tmp, prefix='Neighborhood_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Condition1
    tmp = pd.get_dummies(tmp, prefix='Condition1_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Condition2
    tmp = pd.get_dummies(tmp, prefix='Condition2_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BldgType
    tmp = pd.get_dummies(tmp, prefix='BldgType_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.HouseStyle
    tmp = pd.get_dummies(tmp, prefix='HouseStyle_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.OverallQual
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.OverallCond
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.YearBuilt
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.YearRemodAdd
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.RoofStyle
    tmp = pd.get_dummies(tmp, prefix='RoofStyle_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.RoofMatl
    tmp = pd.get_dummies(tmp, prefix='RoofMatl_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Exterior1st
    tmp = pd.get_dummies(tmp, prefix='Exterior1st_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Exterior2nd
    tmp = pd.get_dummies(tmp, prefix='Exterior2nd_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.MasVnrType
    tmp = pd.get_dummies(tmp, prefix='MasVnrType_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.MasVnrArea
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.ExterQual
    tmp = tmp.map({'Po': 0, 'Fa': 1, 'Ta':2, 'Gd':3, 'Ex':4})
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.ExterCond
    tmp = tmp.map({'Po': 0, 'Fa': 1, 'Ta':2, 'Gd':3, 'Ex':4})
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Foundation
    tmp = pd.get_dummies(tmp, prefix='Foundation_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BsmtQual
    tmp = tmp.map({'Po': 1, 'Fa': 2, 'Ta':3, 'Gd':4, 'Ex':5})
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BsmtCond
    tmp = tmp.map({'Po': 1, 'Fa': 2, 'Ta':3, 'Gd':4, 'Ex':5})
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BsmtExposure
    tmp = tmp.map({'No': 1, 'Mn': 2, 'Av':3, 'Gd':4})
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BsmtFinType1
    tmp = tmp.map({'Unf': 1, 'LwQ': 2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BsmtFinSF1
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BsmtFinType2
    tmp = tmp.map({'Unf': 1, 'LwQ': 2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BsmtFinSF2
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BsmtUnfSF
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.TotalBsmtSF
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Heating
    tmp = pd.get_dummies(tmp, prefix='Heating_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.HeatingQC
    tmp = tmp.map({'Po': 0, 'Fa': 1, 'TA':2, 'Gd':3, 'Ex':4})
    tmp = tmp.fillna(tmp.mean)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.CentralAir
    tmp = tmp.map({'N': 0, 'Y': 1})
    tmp = tmp = tmp.fillna(tmp.mean)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Electrical
    tmp = pd.get_dummies(tmp, prefix='Electrical_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data['1stFlrSF']
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data['2ndFlrSF']
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.LowQualFinSF
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.GrLivArea
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BsmtFullBath
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BsmtHalfBath
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.FullBath
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.HalfBath
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.BedroomAbvGr
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.KitchenAbvGr
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.KitchenQual
    tmp = tmp.map({'Po': 0, 'Fa': 1, 'TA':2, 'Gd':3, 'Ex':4})
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.TotRmsAbvGrd
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Functional
    tmp = tmp.map({'Sal': 0, 'Sev': 1, 'Maj2':2, 'Maj1':3, 'Mod':4,'Min2':5, 'Min1':6, 'Typ':7})
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Fireplaces
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.FireplaceQu
    tmp = tmp.map({'Po': 1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex':5})
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.GarageType
    tmp = pd.get_dummies(tmp, prefix='GarageType_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.GarageYrBlt
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.GarageFinish
    tmp = tmp.map({'Unf': 1, 'RFn':2, 'Fin':3})
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.GarageCars
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.GarageArea
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.GarageQual
    tmp = tmp.map({'Po': 1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex':5})
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.GarageCond
    tmp = tmp.map({'Po': 1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex':5})
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.PavedDrive
    tmp = tmp.map({'N': 0, 'P': 1, 'Y':2})
    tmp = tmp.fillna(tmp.mean)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.WoodDeckSF
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.OpenPorchSF
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.EnclosedPorch
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data['3SsnPorch']
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.ScreenPorch
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.PoolArea
    tmp = tmp.fillna(tmp.mean())
    X = pd.concat([X, tmp], axis=1)

    tmp = data.PoolQC
    tmp = tmp.map({'Fa': 1, 'TA':2, 'Gd':3, 'Ex':4})
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.Fence
    tmp = tmp.map({'MnWw': 1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4})
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.MiscFeature
    tmp = pd.get_dummies(tmp, prefix='MiscFeature_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.MiscVal
    tmp = tmp.fillna(0)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.MoSold
    tmp = tmp.fillna(tmp.mean)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.YrSold
    tmp = tmp.fillna(tmp.mean)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.SaleType
    tmp = pd.get_dummies(tmp, prefix='SaleType_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    tmp = data.SaleCondition
    tmp = pd.get_dummies(tmp, prefix='SaleCondition_', dummy_na=True)
    X = pd.concat([X, tmp], axis=1)

    X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']

    imp = pd.read_csv('feature_importances.csv')
    imp = imp[:90].feature

    for x in X.columns.copy():
        if x not in list(imp):
            #if x in X.columns:
                del X[x]

    # for x in X.columns.copy():
    #     if abs(skew(X[x])) > 0.75:
    #         X[x] = boxcox1p(X[x], 0.15)

    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)

    X_train = X[:train_shape[0]]
    y_train = np.log1p(data_train.SalePrice)
    X_test = X[train_shape[0]:]

    id = np.array(data_test['Id']).reshape(-1, 1)

    return X_train, y_train, X_test, id

def data_val():
    X_train, y_train, X_test, id = load_data()
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.3)
    return X_t, X_v, y_t, y_v