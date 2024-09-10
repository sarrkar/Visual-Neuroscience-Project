import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

data = pd.read_csv('person_1_data.csv')

data['Phase'] = data['Block'].apply(lambda x: 'Phase 1' if x <= 2 else ('Phase 3' if x > 6 else 'Phase 2'))

grouped_phase = data.groupby('Phase')

mean_accuracy = grouped_phase['Correct'].mean()
mean_reaction_time = grouped_phase['Response Time (ms)'].mean()

t_test_results = {
    'Accuracy': {
        'Phase 1 vs Phase 2': ttest_ind(data[data['Phase'] == 'Phase 1']['Correct'], data[data['Phase'] == 'Phase 2']['Correct']),
        'Phase 1 vs Phase 3': ttest_ind(data[data['Phase'] == 'Phase 1']['Correct'], data[data['Phase'] == 'Phase 3']['Correct']),
        'Phase 2 vs Phase 3': ttest_ind(data[data['Phase'] == 'Phase 2']['Correct'], data[data['Phase'] == 'Phase 3']['Correct']),
    },
    'Reaction Time': {
        'Phase 1 vs Phase 2': ttest_ind(data[data['Phase'] == 'Phase 1']['Response Time (ms)'], data[data['Phase'] == 'Phase 2']['Response Time (ms)']),
        'Phase 1 vs Phase 3': ttest_ind(data[data['Phase'] == 'Phase 1']['Response Time (ms)'], data[data['Phase'] == 'Phase 3']['Response Time (ms)']),
        'Phase 2 vs Phase 3': ttest_ind(data[data['Phase'] == 'Phase 2']['Response Time (ms)'], data[data['Phase'] == 'Phase 3']['Response Time (ms)']),
    }
}

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.bar(mean_accuracy.index, mean_accuracy, color=['blue', 'green', 'red'])
plt.title('Mean Accuracy Across Phases')
plt.xlabel('Phase')
plt.ylabel('Mean Accuracy')
for phase, height in mean_accuracy.items():
    plt.text(phase, height, f'{height:.2f}', ha='center', va='bottom')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(mean_reaction_time.index, mean_reaction_time, color=['blue', 'green', 'red'])
plt.title('Mean Reaction Time Across Phases')
plt.xlabel('Phase')
plt.ylabel('Mean Reaction Time (ms)')
for phase, height in mean_reaction_time.items():
    plt.text(phase, height, f'{height:.0f}', ha='center', va='bottom')
plt.grid(True)

plt.tight_layout()
plt.show()

print("T-test results for Accuracy:")
for comparison, result in t_test_results['Accuracy'].items():
    print(f"{comparison}: t-statistic = {result.statistic:.3f}, p-value = {result.pvalue:.3f}")

print("\nT-test results for Reaction Time:")
for comparison, result in t_test_results['Reaction Time'].items():
    print(f"{comparison}: t-statistic = {result.statistic:.3f}, p-value = {result.pvalue:.3f}")