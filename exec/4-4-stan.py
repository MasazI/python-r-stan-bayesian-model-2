import numpy as np
import seaborn as sns
import pandas
import mcmc_tools
import matplotlib.pyplot as plt

# ファイルの読み込み
salary = pandas.read_csv('data-salary.txt')

sample_size = len(salary['X'])
Y = salary['Y']
X = salary['X']

age_range = np.arange(23, 61, 1)

# データの先頭5行の確認
stan_data = {
    'N': sample_size,
    'X': X,
    'Y': Y,
    'N_new': len(age_range),
    'X_new': age_range
}

filename = '../model/model4-4'

# 予測含んだサンプリング
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)


df = pandas.DataFrame(mcmc_result['y_new'])

print(df)

df.columns = age_range

qua = [0.025, 0.25, 0.50, 0.75, 0.975]
d_est = pandas.DataFrame()

for i in np.arange(len(df.columns)):
    for qu in qua:
        d_est[qu] = df.quantile(qu)


print(d_est)

x = d_est.index
y1 = d_est[0.025].values
y2 = d_est[0.25].values
y3 = d_est[0.5].values
y4 = d_est[0.75].values
y5 = d_est[0.975].values

plt.fill_between(x,y1,y5,facecolor='blue',alpha=0.1)
plt.fill_between(x,y2,y4,facecolor='blue',alpha=0.5)
plt.plot(x,y3,'k-')
plt.scatter(salary["X"],salary["Y"],c='b')
plt.show()
