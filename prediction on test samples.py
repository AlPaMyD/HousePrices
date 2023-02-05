import pandas as pd
import pickle

house_price_data = pd.read_csv('test.csv', sep=',')
test_data = pd.read_csv('handled_test.csv', sep=',')

# Test the model
with open('Gradient Boosting regressor.pkl', 'rb') as fid:
    rgr = pickle.load(fid)
pred_SalePrice = rgr.predict(test_data)
pred_SalePrice_df = pd.Series(pred_SalePrice, name='SalePrice')
Id = house_price_data['Id']
prediction_df = pd.concat([Id, pred_SalePrice_df], axis=1)
prediction_df.to_csv('Gradient Boosting regressor prediction to submit.csv', index=False)
print(prediction_df.head())
