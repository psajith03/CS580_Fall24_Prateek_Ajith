import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('linear_regression_data.csv', header=None)
data.columns = ['X', 'Y']

mean_X, mean_Y = np.mean(data['X']), np.mean(data['Y'])
covariance = np.sum((data['X'] - mean_X) * (data['Y'] - mean_Y))
variance = np.sum((data['X'] - mean_X) ** 2)

slope = covariance / variance
intercept = mean_Y - slope * mean_X

plt.scatter(data['X'], data['Y'], color='red')
plt.plot(data['X'], slope * data['X'] + intercept, color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.show()
