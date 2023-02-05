import pandas as pd
import numpy as np


house_price_data = pd.read_csv('train.csv', sep=',')

# Clear the data
# Fill the missing
house_price_data['Electrical'] = house_price_data['Electrical'].replace(np.nan, 'Mix')
house_price_data['GarageYrBlt'] = house_price_data['GarageYrBlt'].replace(np.nan,
                                                                          int(house_price_data['GarageYrBlt'].mean()))
house_price_data = house_price_data.replace(np.nan, 0)

# Map the labels
columns_to_map = ['MSZoning',
                  'Street',
                  'Alley',
                  'LotShape',
                  'LotConfig',
                  'Neighborhood',
                  'Condition1',
                  'Condition2',
                  'BldgType',
                  'HouseStyle',
                  'RoofStyle',
                  'RoofMatl',
                  'Exterior1st',
                  'Exterior2nd',
                  'MasVnrType',
                  'Foundation',
                  'Heating',
                  'CentralAir',
                  'Electrical',
                  'Functional',
                  'GarageType',
                  'PavedDrive',
                  'MiscFeature',
                  'SaleType',
                  'SaleCondition']
for column in columns_to_map:
    column_data = house_price_data[column]
    unique_values = column_data.unique()
    categories = {unique_values: i for i, unique_values in enumerate(unique_values)}
    house_price_data[column] = column_data.map(categories)

columns_to_map_by_hand = ['LandContour',
                          'Utilities',
                          'LandSlope',
                          'ExterQual',
                          'ExterCond',
                          'BsmtQual',
                          'BsmtCond',
                          'BsmtExposure',
                          'BsmtFinType1',
                          'BsmtFinType2',
                          'HeatingQC',
                          'KitchenQual',
                          'FireplaceQu',
                          'GarageFinish',
                          'GarageQual',
                          'GarageCond',
                          'PoolQC',
                          'Fence']

LandContourCategories = {'Lvl': 3,
                         'Bnk': 2,
                         'HLS': 1,
                         'Low': 0}
house_price_data['LandContour'] = house_price_data['LandContour'].map(LandContourCategories)

UtilitiesCategories = {'AllPub': 3,
                       'NoSewr': 2,
                       'NoSeWa': 1,
                       'ELO': 0}
house_price_data['Utilities'] = house_price_data['Utilities'].map(UtilitiesCategories)

LandSlopeCategories = {'Gtl': 2,
                       'Mod': 1,
                       'Low': 0}
house_price_data['LandSlope'] = house_price_data['LandSlope'].map(LandSlopeCategories)

ExterQualCategories = {'Ex': 5,
                       'Gd': 4,
                       'TA': 3,
                       'Fa': 2,
                       'Po': 1}
house_price_data['ExterQual'] = house_price_data['ExterQual'].map(ExterQualCategories)

ExterCondCategories = {'Ex': 5,
                       'Gd': 4,
                       'TA': 3,
                       'Fa': 2,
                       'Po': 1}
house_price_data['ExterCond'] = house_price_data['ExterCond'].map(ExterCondCategories)

BsmtQualCategories = {'Ex': 5,
                      'Gd': 4,
                      'TA': 3,
                      'Fa': 2,
                      'Po': 1}
house_price_data['BsmtQual'] = house_price_data['BsmtQual'].map(BsmtQualCategories)

BsmtCondCategories = {'Ex': 5,
                      'Gd': 4,
                      'TA': 3,
                      'Fa': 2,
                      'Po': 1}
house_price_data['BsmtCond'] = house_price_data['BsmtCond'].map(BsmtCondCategories)

BsmtExposureCategories = {'Gd': 4,
                          'Av': 3,
                          'Mn': 2,
                          'No': 1}
house_price_data['BsmtExposure'] = house_price_data['BsmtExposure'].map(BsmtExposureCategories)

BsmtFinType1Categories = {'GLQ': 6,
                          'ALQ': 5,
                          'BLQ': 4,
                          'Rec': 3,
                          'LwQ': 2,
                          'Unf': 1}
house_price_data['BsmtFinType1'] = house_price_data['BsmtFinType1'].map(BsmtFinType1Categories)

BsmtFinType2Categories = {'GLQ': 6,
                          'ALQ': 5,
                          'BLQ': 4,
                          'Rec': 3,
                          'LwQ': 2,
                          'Unf': 1}
house_price_data['BsmtFinType2'] = house_price_data['BsmtFinType2'].map(BsmtFinType2Categories)

HeatingQCCategories = {'Ex': 5,
                       'Gd': 4,
                       'TA': 3,
                       'Fa': 2,
                       'Po': 1}
house_price_data['HeatingQC'] = house_price_data['HeatingQC'].map(HeatingQCCategories)

KitchenQualCategories = {'Ex': 5,
                         'Gd': 4,
                         'TA': 3,
                         'Fa': 2,
                         'Po': 1}
house_price_data['KitchenQual'] = house_price_data['KitchenQual'].map(KitchenQualCategories)

FireplaceQuCategories = {'Ex': 5,
                         'Gd': 4,
                         'TA': 3,
                         'Fa': 2,
                         'Po': 1}
house_price_data['FireplaceQu'] = house_price_data['FireplaceQu'].map(FireplaceQuCategories)

GarageFinishCategories = {'Fin': 3,
                          'RFn': 2,
                          'Unf': 1}
house_price_data['GarageFinish'] = house_price_data['GarageFinish'].map(GarageFinishCategories)

GarageQualCategories = {'Ex': 5,
                        'Gd': 4,
                        'TA': 3,
                        'Fa': 2,
                        'Po': 1}
house_price_data['GarageQual'] = house_price_data['GarageQual'].map(GarageQualCategories)

GarageCondCategories = {'Ex': 5,
                        'Gd': 4,
                        'TA': 3,
                        'Fa': 2,
                        'Po': 1}
house_price_data['GarageCond'] = house_price_data['GarageCond'].map(GarageCondCategories)

PoolQCCategories = {'Ex': 5,
                    'Gd': 4,
                    'TA': 3,
                    'Fa': 2,
                    'Po': 1}
house_price_data['PoolQC'] = house_price_data['PoolQC'].map(PoolQCCategories)

FenceCategories = {'GdPrv': 4,
                   'MnPrv': 3,
                   'GdWo': 2,
                   'MnWw': 1}
house_price_data['Fence'] = house_price_data['Fence'].map(FenceCategories)

house_price_data = house_price_data.replace(np.nan, 0)

house_price_data = house_price_data.drop(['Id'], axis=1)
house_price_data.to_csv('handled_train.csv', index=False)

# Logarithming the data
columns_to_log = []

columns_to_log.append(house_price_data['LotFrontage'])
columns_to_log.append(house_price_data['LotArea'])
columns_to_log.append(house_price_data['MasVnrArea'])
columns_to_log.append(house_price_data['BsmtFinSF1'])
columns_to_log.append(house_price_data['BsmtUnfSF'])
columns_to_log.append(house_price_data['TotalBsmtSF'])
columns_to_log.append(house_price_data['1stFlrSF'])
columns_to_log.append(house_price_data['2ndFlrSF'])
columns_to_log.append(house_price_data['GrLivArea'])
columns_to_log.append(house_price_data['GarageArea'])
columns_to_log.append(house_price_data['WoodDeckSF'])
columns_to_log.append(house_price_data['OpenPorchSF'])
columns_to_log.append(house_price_data['PoolArea'])
columns_to_log.append(house_price_data['Fence'])
columns_to_log.append(house_price_data['MiscVal'])

columns_to_log = pd.concat(columns_to_log, axis=1)

log_columns_to_log = np.log(columns_to_log + 1)
for column in log_columns_to_log:
    house_price_data[column] = log_columns_to_log[column]

house_price_data.to_csv('log_handled_train.csv', index=False)
