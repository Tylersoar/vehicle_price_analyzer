import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

df = pd.read_csv('datasets/car data.csv')

# finds the max year of the dataset and uses it to calculate the age of the car
reference_year = df['Year'].max()
df['Car Age'] = reference_year - df['Year']
df = df.drop(columns=['Year'])

# filters outliers
Q1 = df['Kms_Driven'].quantile(0.25)
Q3 = df['Kms_Driven'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# removes outliers and X_data takes on more features for sklearns linearregession
df_cleaned = df[(df['Kms_Driven'] >= lower_bound) & (df['Kms_Driven'] <= upper_bound)].reset_index(drop=True)
features = ['Car Age', 'Present_Price', 'Kms_Driven', 'Owner']
x_data = df_cleaned[features]
y_data = df_cleaned['Selling_Price']

# split data up
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# changed linregess to skLearn linearregression to allow to use more features
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# # prints out each features coefficient
for name, coef in zip(features, model.coef_):
    print(f'{name:>15}: {coef:>10.2f}')

# prints out training set coefficient and test set coefficient of determination (R^2), mean absolute error (MAE), and root mean squared error (RMSE)
print("train R^2:", model.score(X_train, y_train))
print("test R^2:", model.score(X_test, y_test))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))

# plotting
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual price (£)')
plt.ylabel('Predicted price (£)')
plt.show()

