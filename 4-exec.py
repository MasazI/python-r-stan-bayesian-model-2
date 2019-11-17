import numpy as np
import pystan
import pandas
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import mcmc_tools

np.random.seed(seed=1234)

n1 = 30
n2 = 20

y1 = np.random.normal(loc=0.0, scale=5.0, size=n1)
y2 = np.random.normal(loc=1.0, scale=4.0, size=n2)

label = []
data = []
for y1_elem in y1:
    label.append('Y1')
    data.append(y1_elem)
for y2_elem in y2:
    label.append('Y2')
    data.append(y2_elem)

df = pandas.DataFrame({'label': label, 'y': data})
print(df.head())

# (1)
sns.violinplot(x='label', y='y', data=df)
plt.show()

stan_data = {
    'y1_N': n1,
    'y2_N': n2,
    'y1': y1,
    'y2': y2
}

filename = '4-exec'

# (2), (3)
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)

# (4)
df = pandas.DataFrame({'mu1': mcmc_result['mu1'], 'mu2': mcmc_result['mu2']})
print(df.head())

df_lower_mu2 = df.query('mu1 < mu2')

prob_lower_mu2 = len(df_lower_mu2)/len(df)

print(prob_lower_mu2)

# (5)
filename_2 = '4-exec-2'
mcmc_result = mcmc_tools.sampling(filename_2, stan_data, n_jobs=4)

df = pandas.DataFrame({'mu1': mcmc_result['mu1'], 'mu2': mcmc_result['mu2']})
print(df)

df_lower_mu2 = df.query('mu1 < mu2')
print(df_lower_mu2)

prob_lower_mu2 = len(df_lower_mu2)/len(df)

print(prob_lower_mu2)