import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df_results = pd.read_csv('/home/ignacio/PycharmProjects/MLFramework/test/results_estimator_analisis.csv', index_col=0)
print(df_results.shape)
print(df_results.head())

df_results_sorted = df_results.sort_values(['mean_test_score','mean_train_score'], ascending=False)
df_to_plot = df_results_sorted[['mean_test_score','mean_train_score']].head()
print(df_to_plot)

fig, ax = plt.subplots()

ax.plot(list(range(df_to_plot.shape[0])),df_to_plot['mean_test_score'], label='mean test score', marker='o')
ax.plot(list(range(df_to_plot.shape[0])),df_to_plot['mean_train_score'], label='mean train score', marker='o')
ax.set_ylim([0,1])
plt.xlabel('Solution')
plt.ylabel('mean')
plt.legend(loc=0, borderaxespad=0.)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.title('Relation between train and test')
ttl = ax.title
ttl.set_position([.5, 1.05])

plt.show()