import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = pd.read_csv('person_1_data.csv')

data['Phase'] = data['Block'].apply(lambda x: 'Phase 1' if x <= 2 else ('Phase 3' if x > 6 else 'Phase 2'))
data['response'] = data['Correct'].apply(lambda x: 1 if x == 1 else 0)
data['rt'] = data['Response Time (ms)'] / 1000

def ddm_likelihood(rt, response, v, a, t0):
    rt = np.array(rt) - t0
    rt[rt < 0] = 1e-5  

    likelihood = norm.pdf(rt, loc=a / (2 * v), scale=np.sqrt(rt / (v * v)))
    likelihood[response == 0] = norm.pdf(rt[response == 0], loc=-a / (2 * v), scale=np.sqrt(rt[response == 0] / (v * v)))

    return likelihood

def neg_log_likelihood(params, rt, response):
    v, a, t0 = params
    if v <= 0 or a <= 0 or t0 < 0:
        return np.inf 
    likelihood = ddm_likelihood(rt, response, v, a, t0)
    return -np.sum(np.log(likelihood))

phases = ['Phase 1', 'Phase 2', 'Phase 3']
ddm_params = {'a': [], 'v': [], 't': []}

for phase in phases:
    phase_data = data[data['Phase'] == phase]
    rt = phase_data['rt'].values
    response = phase_data['response'].values
    initial_params = [1.0, 1.0, 0.3]
    result = minimize(neg_log_likelihood, initial_params, args=(rt, response), method='Nelder-Mead')
    
    v, a, t0 = result.x
    ddm_params['v'].append(v)
    ddm_params['a'].append(a)
    ddm_params['t'].append(t0)

print(ddm_params)

plt.figure(figsize=(18, 5))

# Drift Rate (v)
plt.subplot(1, 3, 1)
plt.plot(phases, ddm_params['v'], '-o', label='Drift Rate')
plt.title('Drift Rate Across Phases')
plt.xlabel('Phase')
plt.ylabel('Drift Rate (v)')
plt.grid(True)

# Decision Bound (a)
plt.subplot(1, 3, 2)
plt.plot(phases, ddm_params['a'], '-o', label='Decision Bound')
plt.title('Decision Bound Across Phases')
plt.xlabel('Phase')
plt.ylabel('Decision Bound (a)')
plt.grid(True)

# Non-decision Time (t)
plt.subplot(1, 3, 3)
plt.plot(phases, ddm_params['t'], '-o', label='Non-decision Time')
plt.title('Non-decision Time Across Phases')
plt.xlabel('Phase')
plt.ylabel('Non-decision Time (t)')
plt.grid(True)

plt.tight_layout()
plt.show()


#Or we can use the code below that uses HDDM:

data = pd.read_csv('person_1_data.csv')
data['Phase'] = data['Block'].apply(lambda x: 'Phase 1' if x <= 2 else ('Phase 3' if x > 6 else 'Phase 2'))
data['response'] = data['Correct'].apply(lambda x: 1 if x == 1 else 0)
data['rt'] = data['Response Time (ms)'] / 1000
phases = ['Phase 1', 'Phase 2', 'Phase 3']

data_hddm = data.rename(columns={'Driving Power':'coherence',
                                 'Response Time (ms)':'rt',
                                 'Correct':'response',
                                 'Phase':'phase'})

def ddm_log_likelihood(params, data):
    v, a, t = params
    rt = data['rt'].values
    response = data['response'].values
    likelihood = np.log(a) - np.log(rt) + (a*v - np.log(np.cosh(a*v)) - 0.5 * (v**2) * rt)
    likelihood[response == 0] = -likelihood[response == 0] 
    return -np.sum(likelihood)

def fit_ddm_for_phase(data_phase):
    initial_params = [0.1, 1.0, 0.3]
    result = minimize(ddm_log_likelihood, initial_params, args=(data_phase,))
    return result.x

ddm_params = {phase: fit_ddm_for_phase(data_hddm[data_hddm['phase'] == phase]) for phase in phases}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, (param_name, param_index, ylabel) in enumerate([('Drift Rate', 0, 'Drift Rate'), ('Decision Bound', 1, 'Decision Bound'), ('Non Dec. time', 2, 'Non Dec. time')]):
    param_values = [ddm_params[phase][param_index] for phase in phases]
    axes[i].plot(phases, param_values, 'o-')
    axes[i].set_xlabel('Phase')
    axes[i].set_ylabel(ylabel)
    axes[i].set_xticks(phases)
    axes[i].set_xticklabels(phases)

plt.tight_layout()
plt.show()