import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('datasets/car data.csv')

x_data = df['Year']
y_data = df['Selling_Price']

Q1 = df['Kms_Driven'].quantile(0.25)
Q3 = df['Kms_Driven'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df[(df['Kms_Driven'] >= lower_bound) & (df['Kms_Driven'] <= upper_bound)].reset_index(drop=True)


df_cleaned.plot(x='Year', y='Selling_Price', kind='scatter', c='Kms_Driven', cmap='Greens',
        title='year vs Selling Price', xlabel='Year', ylabel='Selling Price')

plt.gcf().axes[1].set_ylabel('Milage')

plt.show()

