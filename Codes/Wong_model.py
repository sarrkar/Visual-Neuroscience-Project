import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

def wong_model(thr0, µ, coh12, JN11, JN0, I, timesteps=2000, dt=0.001):
    reaction_times = []
    accuracies = []
    
    for _ in range(10000):
        x = 0
        t = 0
        for t in range(timesteps):
            dx = (µ + coh12 * x + JN11 * x - JN0 * x + I) * dt
            x += dx
            
            if x >= thr0:
                reaction_times.append(t * dt)
                accuracies.append(1)
                break
            elif x <= -thr0:
                reaction_times.append(t * dt)
                accuracies.append(0)
                break
        else:
            reaction_times.append(timesteps * dt)
            accuracies.append(0.5)
    
    return reaction_times, accuracies

data = pd.read_csv('person_1_data.csv')
data['Phase'] = data['Block'].apply(lambda x: 'Phase 1' if x <= 2 else ('Phase 3' if x > 6 else 'Phase 2'))
data['response'] = data['Correct'].apply(lambda x: 1 if x == 1 else 0)
data['rt'] = data['Response Time (ms)'] / 1000 

data_phase1 = data[data['Phase'] == 'Phase 1']
data_phase3 = data[data['Phase'] == 'Phase 3']

def cost_function(params, data):
    thr0, µ = params
    reaction_times, accuracies = wong_model(thr0, µ, coh12=0.5, JN11=0.5, JN0=0.1, I=0.1)
    rt_hist, rt_bins = np.histogram(reaction_times, bins=30, density=True)
    data_hist, data_bins = np.histogram(data['rt'], bins=30, density=True)
    accuracy_diff = np.mean(np.abs(np.array(accuracies) - data['response'].mean()))
    rt_diff = np.sum((rt_hist - data_hist) ** 2)
    return accuracy_diff + rt_diff

initial_params = [0.5, 0.1]
result_phase1 = minimize(cost_function, initial_params, args=(data_phase1,), method='Nelder-Mead')
params_phase1 = result_phase1.x

result_phase3 = minimize(cost_function, initial_params, args=(data_phase3,), method='Nelder-Mead')
params_phase3 = result_phase3.x

plt.figure(figsize=(12, 5))

# Plot reaction time distributions
reaction_times_phase1, _ = wong_model(params_phase1[0], params_phase1[1], coh12=0.5, JN11=0.5, JN0=0.1, I=0.1)
reaction_times_phase3, _ = wong_model(params_phase3[0], params_phase3[1], coh12=0.5, JN11=0.5, JN0=0.1, I=0.1)

plt.subplot(1, 2, 1)
plt.hist(reaction_times_phase1, bins=30, alpha=0.5, label='Phase 1')
plt.hist(data_phase1['rt'], bins=30, alpha=0.5, label='Data Phase 1')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Frequency')
plt.title('Reaction Time Distribution - Phase 1')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(reaction_times_phase3, bins=30, alpha=0.5, label='Phase 3')
plt.hist(data_phase3['rt'], bins=30, alpha=0.5, label='Data Phase 3')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Frequency')
plt.title('Reaction Time Distribution - Phase 3')
plt.legend()

plt.tight_layout()
plt.show()

# Plot accuracy comparison
plt.figure(figsize=(6, 4))
accuracies_phase1 = np.mean(wong_model(params_phase1[0], params_phase1[1], coh12=0.5, JN11=0.5, JN0=0.1, I=0.1)[1])
accuracies_phase3 = np.mean(wong_model(params_phase3[0], params_phase3[1], coh12=0.5, JN11=0.5, JN0=0.1, I=0.1)[1])
plt.bar(['Phase 1', 'Phase 3'], [accuracies_phase1, accuracies_phase3], alpha=0.5, label='Model')
plt.bar(['Phase 1', 'Phase 3'], [data_phase1['response'].mean(), data_phase3['response'].mean()], alpha=0.5, label='Data')
plt.xlabel('Phase')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend()
plt.show()
