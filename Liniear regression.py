from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import quandl
import math
import matplotlib.pyplot as plt

import datetime


df = quandl.get('WIKI/GOOGL')

df = df[["Adj. Open", "Adj. Low", "Adj. Close", "Adj. Volume", "Adj. High"]]  # Pull features we want
# Creating new features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)  # Make NAN values outliers

forecast_out = int(input())  # Number of days out we forecast

df['label'] = df[forecast_col].shift(-forecast_out)  # Shifting columns negatively

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x_latest = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

# train test split
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

# fit with linear regression
clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)  # Squared error
print("Linear Regression accuracy", accuracy)
forecast_set = clf.predict(x_latest)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for date in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for i in range(len(df.columns)-1)] + [date]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
