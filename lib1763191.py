from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score

def cv(regr, train_features, train_prices, kf=3):
    '''
    Calculate a k-fold cross validation
    :param regr: model to apply
    :param train_features: features numpy matrix
    :param train_prices: prices numpy vector
    :param kf: how many folds (by default 3)
    :return:
    '''
    print("Calculating CV...", end="")
    k_fold = KFold(len(train_prices), n_folds=kf, shuffle=True, random_state=0)
    cv = cross_val_score(regr, train_features, train_prices, cv=k_fold, n_jobs=-1)
    print("   [DONE]")
    print("\n--- CV with %s fold" % kf)
    print("---- Mean: %s" % np.mean(cv))
    print("----  STD: %s\n" % np.std(cv))
    return 0


def encode(col_name, data):
    '''
    Maps the column provided from string to int
    :param col_name: name of the columns to encode
    :param data: dataframe
    :return: same dataframe with the col_name encoded to numeric values
    '''
    le = preprocessing.LabelEncoder()
    data[col_name] = le.fit_transform(data[col_name].values)
    return data


def eval_clean(col_name, data):
    '''
    :param col_name: name of the column
    :param data: data matrix (ie training + test set)
    :return: data matrix with the col_name mapped to 0-5 and with no Na
    '''
    data.loc[:, col_name] = data.loc[:, col_name].fillna("TA")
    return data.replace({col_name: {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}})


def clean(data):
    '''
    Cleans the data provided as a pd data frame
    :param data: pandas data frame
    :return: the same data frame but cleaned
    '''
    # Edit the following Columns
    # Fill with zero when the house lacks the numerical feature
    # or with the "normal" value when is categorical
    print("Cleaning the features...", end="", flush=True)
    data.loc[:, "MSSubClass"] = data.loc[:, "MSSubClass"].fillna(0)
    data.loc[:, "MSZoning"] = data.loc[:, "MSZoning"].fillna("No")
    data = encode("MSZoning", data)
    data.loc[:, "LotFrontage"] = data.loc[:, "LotFrontage"].fillna(0)
    data.loc[:, "LotArea"] = data.loc[:, "LotArea"].fillna(data["LotArea"].median(axis=0))
    data.loc[:, "Street"] =  data.loc[:, "Street"].fillna("Pave")
    data = encode("Street", data)
    data.loc[:, "Alley"] = data.loc[:, "Alley"].fillna("No")
    data = encode("Alley", data)
    data.loc[:, "LotShape"] = data.loc[:, "LotShape"].fillna(data["LotShape"].mode().values[0])
    data = data.replace({"LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4}})
    data.loc[:, "LandContour"] = data.loc[:, "LandContour"].fillna(data["LandContour"].mode().values[0])
    data = encode("LandContour", data)
    data.loc[:, "Utilities"] = data.loc[:, "Utilities"].fillna("AllPub")
    data = data.replace({"Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}})
    data.loc[:, "LotConfig"] = data.loc[:, "LotConfig"].fillna(data["LotConfig"].mode().values[0])
    data = encode("LotConfig", data)
    data.loc[:, "LandSlope"] = data.loc[:, "LandSlope"].fillna(data["LandSlope"].mode().values[0])
    data = data.replace({"LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3}})
    data.loc[:, "Neighborhood"] = data.loc[:, "Neighborhood"].fillna(data["Neighborhood"].mode().values[0])
    data = encode("Neighborhood", data)
    data.loc[:, "Condition1"] = data.loc[:, "Condition1"].fillna("Norm")
    data = encode("Condition1", data)
    data.loc[:, "Condition2"] = data.loc[:, "Condition2"].fillna("Norm")
    data = encode("Condition2", data)
    data.loc[:, "BldgType"] = data.loc[:, "BldgType"].fillna(data["BldgType"].mode().values[0])
    data = encode("BldgType", data)
    data.loc[:, "HouseStyle"] = data.loc[:, "HouseStyle"].fillna(data["HouseStyle"].mode().values[0])
    data = encode("HouseStyle", data) # Change to replace
    data.loc[:, "OverallQual"] = data.loc[:, "OverallQual"].fillna(5)
    data.loc[:, "OverallCond"] = data.loc[:, "OverallCond"].fillna(5)
    yb_mode = data["YearBuilt"].mode().values[0]
    data.loc[:, "YearBuilt"] = data.loc[:, "YearBuilt"].fillna(yb_mode)
    data.loc[:, "YearRemodAdd"] = data.loc[:, "YearRemodAdd"].fillna(yb_mode)
    data.loc[:, "RoofStyle"] = data.loc[:, "RoofStyle"].fillna(data["RoofStyle"].mode().values[0])
    data = encode("RoofStyle", data)
    data.loc[:, "RoofMatl"] = data.loc[:, "RoofMatl"].fillna(data["RoofMatl"].mode().values[0])
    data = encode("RoofMatl", data)
    data.loc[:, "Exterior1st"] = data.loc[:, "Exterior1st"].fillna(data["Exterior1st"].mode().values[0])
    data = encode("Exterior1st", data)
    data.loc[:, "Exterior2nd"] = data.loc[:, "Exterior2nd"].fillna("No")
    data = encode("Exterior2nd", data)
    data.loc[:, "MasVnrType"] = data.loc[:, "MasVnrType"].fillna("No")
    data = encode("MasVnrType", data)
    data.loc[:, "MasVnrArea"] = data.loc[:, "MasVnrArea"].fillna(0)
    data = eval_clean("ExterQual", data)
    data = eval_clean("ExterCond", data)
    data.loc[:, "Foundation"] = data.loc[:, "Foundation"].fillna(data["Foundation"].mode().values[0])
    data = encode("Foundation", data)
    data = eval_clean("BsmtQual", data)
    data = eval_clean("BsmtCond", data)
    data.loc[:, "BsmtExposure"] = data.loc[:, "BsmtExposure"].fillna("Zero")
    data = data.replace({"BsmtExposure": {"Zero": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}})
    data.loc[:, "BsmtFinType1"] = data.loc[:, "BsmtFinType1"].fillna("No")
    data = data.replace({"BsmtFinType1": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}})
    data.loc[:, "BsmtFinSF1"] = data.loc[:, "BsmtFinSF1"].fillna(0)
    data.loc[:, "BsmtFinType2"] = data.loc[:, "BsmtFinType2"].fillna("No")
    data = data.replace({"BsmtFinType2": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}})
    data.loc[:, "BsmtFinSF2"] = data.loc[:, "BsmtFinSF2"].fillna(0)
    data.loc[:, "BsmtUnfSF"] = data.loc[:, "BsmtUnfSF"].fillna(0)
    data.loc[:, "TotalBsmtSF"] = data.loc[:, "TotalBsmtSF"].fillna(0)
    data.loc[:, "Heating"] = data.loc[:, "Heating"].fillna("No")
    data = encode("Heating", data)
    data.loc[:, "HeatingQC"] = data.loc[:, "HeatingQC"].fillna("No")
    data = eval_clean("HeatingQC", data)
    data.loc[:, "CentralAir"] = data.loc[:, "CentralAir"].fillna("N")
    data = data.replace({"CentralAir":{"Y": 1, "N": 0}})
    data.loc[:, "Electrical"] = data.loc[:, "Electrical"].fillna("FuseA")
    data = data.replace({"Electrical": {"FuseP": 0, "FuseF": 1, "FuseA": 2, "SBrkr": 3, "Mix": 2.7}})
    data.loc[:, "1stFlrSF"].fillna(data["1stFlrSF"].median(axis=0), inplace=True)
    data.loc[:, "2ndFlrSF"].fillna(data["2ndFlrSF"].median(axis=0), inplace=True)
    data.loc[:, "LowQualFinSF"].fillna(data["LowQualFinSF"].median(axis=0), inplace=True)
    data.loc[:, "GrLivArea"].fillna(data["GrLivArea"].median(axis=0), inplace=True)
    data.loc[:, "BsmtFullBath"].fillna(data["BsmtFullBath"].median(axis=0), inplace=True)
    data.loc[:, "BsmtFullBath"] = data.loc[:, "BsmtFullBath"].fillna(0)
    data.loc[:, "BsmtHalfBath"] = data.loc[:, "BsmtHalfBath"].fillna(0)
    data.loc[:, "FullBath"] = data.loc[:, "FullBath"].fillna(0)
    data.loc[:, "HalfBath"] = data.loc[:, "HalfBath"].fillna(0)
    data.loc[:, "KitchenQual"] = data.loc[:, "KitchenQual"].fillna("No")
    data = eval_clean("KitchenQual", data)
    data.loc[:, "TotRmsAbvGrd"] = data.loc[:, "TotRmsAbvGrd"].fillna(0)
    data.loc[:, "Functional"] = data.loc[:, "Functional"].fillna("Typ")
    data = data.replace({"Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                                          "Min2": 6, "Min1": 7, "Typ": 8}})
    data.loc[:, "Fireplaces"] = data.loc[:, "Fireplaces"].fillna(0)
    data.loc[:, "FireplaceQu"] = data.loc[:, "FireplaceQu"].fillna("No")
    data = eval_clean("FireplaceQu", data)
    data.loc[:, "GarageType"] = data.loc[:, "GarageType"].fillna("No")
    data = data.replace({"GarageType":{"No": 0, "Detchd": 1, "CarPort": 2, "BuiltIn": 3, "Basment": 4, "Attchd": 5, "2Types": 6}})
    data.loc[:, "GarageYrBlt"] = data.loc[:, "GarageYrBlt"].fillna(0)
    data.loc[:, "GarageFinish"] = data.loc[:, "GarageFinish"].fillna("No")
    data = data.replace({"GarageFinish": {"No": 0, "Unf": 1, "RFn": 2, "Fin": 3}})
    data.loc[:, "GarageCars"] = data.loc[:, "GarageCars"].fillna(0)
    data.loc[:, "GarageArea"] = data.loc[:, "GarageArea"].fillna(0)
    data.loc[:, "GarageQual"] = data.loc[:, "GarageQual"].fillna("No")
    data = eval_clean("GarageQual", data)
    data.loc[:, "GarageCond"] = data.loc[:, "GarageCond"].fillna("No")
    data = eval_clean("GarageCond", data)
    data.loc[:, "PavedDrive"] = data.loc[:, "PavedDrive"].fillna("No")
    data = data.replace({"PavedDrive": {"No": 0, "N": 1, "P": 2, "Y": 3}})
    data.loc[:, "WoodDeckSF"] = data.loc[:, "WoodDeckSF"].fillna(0)
    data.loc[:, "OpenPorchSF"] = data.loc[:, "OpenPorchSF"].fillna(0)
    data.loc[:, "EnclosedPorch"] = data.loc[:, "EnclosedPorch"].fillna(0)
    data.loc[:, "3SsnPorch"] = data.loc[:, "3SsnPorch"].fillna(0)
    data.loc[:, "ScreenPorch"] = data.loc[:, "ScreenPorch"].fillna(0)
    data.loc[:, "PoolArea"] = data.loc[:, "PoolArea"].fillna(0)
    data.loc[:, "PoolQC"] = data.loc[:, "PoolQC"].fillna("No")
    data = eval_clean("PoolQC", data)
    data.loc[:, "Fence"] = data.loc[:, "Fence"].fillna("No")
    data = data.replace({"Fence": {"No": 0, "MnWw": 1, "GdWo":2, "MnPrv":3, "GdPrv":4}})
    data.loc[:, "MiscFeature"] = data.loc[:, "MiscFeature"].fillna("No")
    data = encode("MiscFeature", data)
    data = encode("MiscFeature", data)
    data.loc[:, "MiscVal"] = data.loc[:, "MiscVal"].fillna(0)
    data.loc[:, "MoSold"] = data.loc[:, "MoSold"].fillna(0)
    data.loc[:, "YrSold"] = data.loc[:, "YrSold"].fillna(0)
    data.loc[:, "SaleType"] = data.loc[:, "SaleType"].fillna(data["SaleType"].mode().values[0])
    data = encode("SaleType", data) # change to order the values
    data.loc[:, "SaleCondition"] = data.loc[:, "SaleCondition"].fillna("Normal")
    data = encode("SaleCondition", data)
    print("   [DONE]")
    return data


def add_features(data):
    '''
    Here I add new features
    :param data: dataframe to modify
    :return: dataframe modified
    '''
    print("Adding new features...", end="", flush=True)
    # Add relevant features together
    data['sf_floors'] = data['1stFlrSF'] + data['2ndFlrSF']
    data['all_liveable'] = data['sf_floors'] + data['LowQualFinSF'] + data['GrLivArea']

    # Add new columns thats says if there are remodels
    data["Remodel"] = (data.loc[:, "YearBuilt"] == data.loc[:, "YearRemodAdd"]) * 1
    data = data.drop("YearRemodAdd", 1)

    # Check if the house was sold the same year it was built
    data["VeryNewHouse"] = (data["YearBuilt"] == data["YrSold"]) * 1
    print("   [DONE]")
    return data


def zscore(data):
    '''
    Standardize the values of each columns with mean 0 and STD 1
    :param data: dataframe to standardize
    :return: standardized dataframe
    '''
    print("Standardizing the features...", end="", flush=True)
    cols = data.columns
    for col in cols:
        data[col] = (data[col] - data[col].mean())/data[col].std(ddof=0)
    print("   [DONE]")
    return data


def export(ind, prices):
    '''
    :param ind: index of rows
    :param prices: price for each house
    :return: none
    '''
    print("Exporting the results...", end="", flush=True)
    preds = pd.DataFrame({"SalePrice": prices}, index=ind)
    preds.to_csv("preds.csv")  # this can be submitted to Kaggle!
    print("   [DONE]")
