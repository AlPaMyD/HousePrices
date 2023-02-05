import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

house_price_data = pd.read_csv('handled_train.csv', sep=',')

# Pick the columns to be in train data
columns_in_train_data = []

# columns_in_train_data.append(house_price_data['Id'])
columns_in_train_data.append(house_price_data['MSSubClass'])
columns_in_train_data.append(house_price_data['MSZoning'])
columns_in_train_data.append(house_price_data['LotFrontage'])
columns_in_train_data.append(house_price_data['LotArea'])
columns_in_train_data.append(house_price_data['Street'])
columns_in_train_data.append(house_price_data['Alley'])
columns_in_train_data.append(house_price_data['LotShape'])
columns_in_train_data.append(house_price_data['LandContour'])
columns_in_train_data.append(house_price_data['Utilities'])
columns_in_train_data.append(house_price_data['LotConfig'])
columns_in_train_data.append(house_price_data['LandSlope'])
columns_in_train_data.append(house_price_data['Neighborhood'])
columns_in_train_data.append(house_price_data['Condition1'])
columns_in_train_data.append(house_price_data['Condition2'])
columns_in_train_data.append(house_price_data['BldgType'])
columns_in_train_data.append(house_price_data['HouseStyle'])
columns_in_train_data.append(house_price_data['OverallQual'])
columns_in_train_data.append(house_price_data['OverallCond'])
columns_in_train_data.append(house_price_data['YearBuilt'])
columns_in_train_data.append(house_price_data['YearRemodAdd'])
columns_in_train_data.append(house_price_data['RoofStyle'])
columns_in_train_data.append(house_price_data['RoofMatl'])
columns_in_train_data.append(house_price_data['Exterior1st'])
columns_in_train_data.append(house_price_data['Exterior2nd'])
columns_in_train_data.append(house_price_data['MasVnrType'])
columns_in_train_data.append(house_price_data['MasVnrArea'])
columns_in_train_data.append(house_price_data['ExterQual'])
columns_in_train_data.append(house_price_data['ExterCond'])
columns_in_train_data.append(house_price_data['Foundation'])
columns_in_train_data.append(house_price_data['BsmtQual'])
columns_in_train_data.append(house_price_data['BsmtCond'])
columns_in_train_data.append(house_price_data['BsmtExposure'])
columns_in_train_data.append(house_price_data['BsmtFinType1'])
columns_in_train_data.append(house_price_data['BsmtFinSF1'])
columns_in_train_data.append(house_price_data['BsmtFinType2'])
columns_in_train_data.append(house_price_data['BsmtFinSF2'])
columns_in_train_data.append(house_price_data['BsmtUnfSF'])
columns_in_train_data.append(house_price_data['TotalBsmtSF'])
columns_in_train_data.append(house_price_data['Heating'])
columns_in_train_data.append(house_price_data['HeatingQC'])
columns_in_train_data.append(house_price_data['CentralAir'])
columns_in_train_data.append(house_price_data['Electrical'])
columns_in_train_data.append(house_price_data['1stFlrSF'])
columns_in_train_data.append(house_price_data['2ndFlrSF'])
columns_in_train_data.append(house_price_data['LowQualFinSF'])
columns_in_train_data.append(house_price_data['GrLivArea'])
columns_in_train_data.append(house_price_data['BsmtFullBath'])
columns_in_train_data.append(house_price_data['BsmtHalfBath'])
columns_in_train_data.append(house_price_data['FullBath'])
columns_in_train_data.append(house_price_data['HalfBath'])
columns_in_train_data.append(house_price_data['BedroomAbvGr'])
columns_in_train_data.append(house_price_data['KitchenAbvGr'])
columns_in_train_data.append(house_price_data['KitchenQual'])
columns_in_train_data.append(house_price_data['TotRmsAbvGrd'])
columns_in_train_data.append(house_price_data['Functional'])
columns_in_train_data.append(house_price_data['Fireplaces'])
columns_in_train_data.append(house_price_data['FireplaceQu'])
columns_in_train_data.append(house_price_data['GarageType'])
columns_in_train_data.append(house_price_data['GarageYrBlt'])
columns_in_train_data.append(house_price_data['GarageFinish'])
columns_in_train_data.append(house_price_data['GarageCars'])
columns_in_train_data.append(house_price_data['GarageArea'])
columns_in_train_data.append(house_price_data['GarageQual'])
columns_in_train_data.append(house_price_data['GarageCond'])
columns_in_train_data.append(house_price_data['PavedDrive'])
columns_in_train_data.append(house_price_data['WoodDeckSF'])
columns_in_train_data.append(house_price_data['OpenPorchSF'])
columns_in_train_data.append(house_price_data['EnclosedPorch'])
columns_in_train_data.append(house_price_data['3SsnPorch'])
columns_in_train_data.append(house_price_data['ScreenPorch'])
columns_in_train_data.append(house_price_data['PoolArea'])
columns_in_train_data.append(house_price_data['PoolQC'])
columns_in_train_data.append(house_price_data['Fence'])
columns_in_train_data.append(house_price_data['MiscFeature'])
columns_in_train_data.append(house_price_data['MiscVal'])
columns_in_train_data.append(house_price_data['MoSold'])
columns_in_train_data.append(house_price_data['YrSold'])
columns_in_train_data.append(house_price_data['SaleType'])
columns_in_train_data.append(house_price_data['SaleCondition'])

# Arrange the train data
train_data = pd.concat(columns_in_train_data, axis=1)

# Split the Data
SalePrice = house_price_data['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(train_data, SalePrice, test_size=0.2)


# Train the model
def train_the_regressor(x_train, x_test, y_train, y_test, training_attempts=10):
    # Train the model
    min_error = 2**10
    for i in range(training_attempts):
        rgr = GradientBoostingRegressor(n_estimators=200, criterion='squared_error')
        rgr.fit(x_train, y_train)
        pred_SalePrice = rgr.predict(x_test)
        train_error = metrics.mean_squared_error(np.log(y_test), np.log(pred_SalePrice))
        if train_error < min_error:
            min_error = train_error
            best_rgr = rgr
    return best_rgr


training_attempts = 100
best_rgr = train_the_regressor(x_train, x_test, y_train, y_test, training_attempts=training_attempts)
pred_SalePrice = best_rgr.predict(x_test)
min_error = metrics.mean_squared_error(np.log(y_test), np.log(pred_SalePrice))

print(f"Min Error: {min_error}")

# save the regressor
with open('Gradient Boosting regressor.pkl', 'wb') as fid:
    pickle.dump(best_rgr, fid)

# Extract important features
feature_importances = list(zip(train_data.columns, best_rgr.feature_importances_ * 100))
filtered_feature_importances = []
names_of_important_features = []
importance_threshold = 1
for column, feature_importance in feature_importances:
    if feature_importance >= importance_threshold:
        filtered_feature_importances.append((column, feature_importance))
        names_of_important_features.append(column)
print(filtered_feature_importances)
print(names_of_important_features)

# Arrange the new train data for important features
columns_in_train_data = [house_price_data[feature_name] for feature_name in names_of_important_features]
train_data_with_no_minor_features = pd.concat(columns_in_train_data, axis=1)
SalePrice = house_price_data['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(train_data_with_no_minor_features, SalePrice, test_size=0.2)

# Train the model
best_rgr = train_the_regressor(x_train, x_test, y_train, y_test, training_attempts=training_attempts)
pred_SalePrice = best_rgr.predict(x_test)
min_error = metrics.mean_squared_error(np.log(y_test), np.log(pred_SalePrice))
print(f"Min Error with important features: {min_error}")

# save the regressor
with open('Gradient Boosting regressor with important features.pkl', 'wb') as fid:
    pickle.dump(best_rgr, fid)
