import pandas as pd

house_price_data = pd.read_csv('train.csv', sep=',')
for column in house_price_data.columns:
    print(f"columns_in_train_data.append(house_price_data['{column}'])")
