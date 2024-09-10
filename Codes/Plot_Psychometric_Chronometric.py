#Psychometric Function and Chronometric Function

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

data = pd.read_csv('person_1_data.csv')

grouped = data.groupby(['Driving Power'])

psychometric_data = grouped['Correct'].mean().reset_index()

chronometric_data = grouped['Response Time (ms)'].mean().reset_index()
response_time_std = grouped['Response Time (ms)'].std().reset_index()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(psychometric_data['Driving Power'], psychometric_data['Correct'], marker='o', label='Proportion Correct')
plt.title('Psychometric Function')
plt.xlabel('Motion Strength (%Coh)')
plt.ylabel('Probability Correct')
plt.ylim(0.5, 1.0)
plt.xlim(0, max(psychometric_data['Driving Power']))
plt.xticks(psychometric_data['Driving Power'])
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.errorbar(chronometric_data['Driving Power'], chronometric_data['Response Time (ms)'],
             yerr=response_time_std['Response Time (ms)'], marker='o', label='Reaction Time')
plt.title('Chronometric Function')
plt.xlabel('Motion Strength (%Coh)')
plt.ylabel('Reaction Time (ms)')
plt.ylim(300, 2000)
plt.xlim(0, max(chronometric_data['Driving Power']))
plt.xticks(chronometric_data['Driving Power'])
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#Psychometric Function and Chronometric Function with Polynomial fit
data = pd.read_csv('person_32_data.csv')

grouped = data.groupby(['Driving Power'])

psychometric_data = grouped['Correct'].mean().reset_index()

chronometric_data = grouped['Response Time (ms)'].mean().reset_index()
response_time_std = grouped['Response Time (ms)'].std().reset_index()

def logistic(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

popt_psycho, _ = curve_fit(logistic, psychometric_data['Driving Power'], psychometric_data['Correct'],
                           bounds=([0.5, 0, 0, 0], [1.5, 100, 1, 1]))
x_fit = np.linspace(0, max(psychometric_data['Driving Power']), 100)
y_fit_psycho = logistic(x_fit, *popt_psycho)

def polynomial(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

popt_chrono, _ = curve_fit(polynomial, chronometric_data['Driving Power'], chronometric_data['Response Time (ms)'])
y_fit_chrono = polynomial(x_fit, *popt_chrono)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(psychometric_data['Driving Power'], psychometric_data['Correct'], 'o', label='Proportion Correct')
plt.plot(x_fit, y_fit_psycho, '-', label='Logistic Fit')
plt.title('Psychometric Function')
plt.xlabel('Motion Strength (%Coh)')
plt.ylabel('Probability Correct')
plt.ylim(0.5, 1.0)
plt.xlim(0, max(psychometric_data['Driving Power']))
plt.xticks(psychometric_data['Driving Power'])
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.errorbar(chronometric_data['Driving Power'], chronometric_data['Response Time (ms)'],
             yerr=response_time_std['Response Time (ms)'], fmt='o', label='Reaction Time')
plt.plot(x_fit, y_fit_chrono, '-', label='Polynomial Fit')
plt.title('Chronometric Function')
plt.xlabel('Motion Strength (%Coh)')
plt.ylabel('Reaction Time (ms)')
plt.ylim(300, 2000)
plt.xlim(0, max(chronometric_data['Driving Power']))
plt.xticks(chronometric_data['Driving Power'])
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
