import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

cols = ['model','year','price','transmission','mileage','fuelType','tax','mpg','engineSize']
df = pd.read_csv('datasets/Used-cars.csv', comment='#', header=None,
                 names=cols, skiprows=4, on_bad_lines='skip')

# put engine size in the tax slot
short = df['engineSize'].isna() & df['tax'].notna()
df.loc[short, 'engineSize'] = df.loc[short, 'tax']
df = df.drop(columns=['tax','mpg'])


for c in ['year','price','mileage','engineSize']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna()
df['model'] = df['model'].str.strip()

current_year = datetime.now().year

df = df[df['engineSize']>0] # 0.0L = missing
df = df[df['transmission'] != 'Other']
df = df[df['year'] <= current_year] # remove car from 2060
df = df.drop_duplicates().reset_index(drop=True)

reference_year = df['year'].max()
df['Car Age'] = df['year'] - reference_year

X = pd.get_dummies(df[['Car Age','mileage','engineSize','model','fuelType','transmission']],
                   columns=['model','fuelType','transmission'],
                   drop_first=True
                   )
y = df['price']

y_log = np.log(y)

X_train, X_test,y_train_log,y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train_log)

y_pred = np.exp(rf.predict(X_test))
y_true = np.exp(y_test_log)

print("test R^2 (£):", r2_score(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("RMSE:", root_mean_squared_error(y_true, y_pred))

plt.scatter(y_true, y_pred, alpha=0.6)
lims = [y_true.min(), y_true.max()]
plt.plot(lims, lims, 'k--')
plt.xlabel('Actual price (£)')
plt.ylabel('Predicted price (£)')
plt.title("Random Forest Regressor: Actual vs Predicted Prices")
plt.show()
