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

# データの先頭5行の確認
stan_data = {
    'N': sample_size,
    'X': X,
    'Y': Y
}

filename = 'model4-5'

mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)

# サンプリングデータの散布図
d_mcmc = pandas.DataFrame({'Intercept': mcmc_result['Intercept'],
                  'b': mcmc_result['b'],
                  'sigma': mcmc_result['sigma']})

print(d_mcmc.head())

sns.jointplot("Intercept", "b", data=d_mcmc, kind="reg", color="m", height=7)
plt.show()

# 50歳の予測分布のサンプリング
age = 50
mean = mcmc_result['Intercept'].mean() + mcmc_result['b'].mean() * age
print(mean)
print(mcmc_result['sigma'].mean())
y50 = np.random.normal(loc=mean, scale=mcmc_result['sigma'].mean(),size=len(mcmc_result['lp__']))

# 23~60歳の予測分布のサンプリングと図示
age_range = np.arange(23, 61, 1)
predictions = []
for i in age_range:
    mean = mcmc_result['Intercept'].mean() + mcmc_result['b'].mean() * i
    y = np.random.normal(loc=mean, scale=mcmc_result['sigma'].mean(), size=len(mcmc_result['lp__']))
    predictions.append(y)

df = pandas.DataFrame(np.array(predictions).T)

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
