import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


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

df_cleaned = df[(df['Kms_Driven'] >= lower_bound) & (df['Kms_Driven'] <= upper_bound)].reset_index(drop=True)
x_data = df_cleaned['Car Age']
y_data = df_cleaned['Selling_Price']

# linear regression
slope, intercept, r, p, std_err = stats.linregress(x_data, y_data)

def myfunc(x):
    return slope * x + intercept

myModel = list(map(myfunc, x_data))

# plotting
ax = df_cleaned.plot(x='Car Age', y='Selling_Price', kind='scatter', c='Kms_Driven', cmap='Greens',
        title='Car Age vs Selling Price', xlabel='Car Age', ylabel='Selling Price')

ax.plot(x_data, myModel, color='green', label=f'Regression Line (R={r:.2f})')

plt.gcf().axes[1].set_ylabel('Milage')

plt.legend(loc='best')
plt.show()

