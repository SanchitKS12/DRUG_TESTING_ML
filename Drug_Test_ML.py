import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')
#print(df)

y=df['logS']
x=df.drop('logS',axis=1) #axis=0 allows it to work in row mode whearas axis=1 allows it to work in coloum mode
#print(x)

from sklearn.model_selection import train_test_split

x_train=train_test_split(x) #Contains 80% of provided data
x_test=train_test_split(y)  #Contains 20% of provided data
y_train=train_test_split(test_size=0.2)
y_test=train_test_split(random_state=100) #random_state :Allows the same amount of data split everytime while processing

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_prediction = lr.predict(x_train)
y_lr_test_prediction = lr.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score #

lr_train_mse = mean_squared_error(y_train, y_lr_train_prediction)
lr_train_r2 = r2_score(y_train, y_lr_train_prediction)

lr_test_mse = mean_squared_error(y_test, y_lr_test_prediction)
lr_test_r2 = r2_score(y_test, y_lr_test_prediction)

lr_finale = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_finale.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

print(lr_finale)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

y_rf_train_prediction = rf.predict(x_train)
y_rf_test_prediction = rf.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_prediction)
rf_train_r2 = r2_score(y_train, y_rf_train_prediction)

rf_test_mse = mean_squared_error(y_test, y_rf_test_prediction)
rf_test_r2 = r2_score(y_test, y_rf_test_prediction)

rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

df_models = pd.concat([lr_finale, rf_results], axis=0)
df_models.reset_index(drop=True)

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_prediction, c="#7CAE00" ,alpha=0.3)

z = np.polyfit(y_train, y_lr_train_prediction, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')

print("Thank you")
