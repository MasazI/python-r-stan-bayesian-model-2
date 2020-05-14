# 練習問題8(6)
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.gridspec import GridSpec
import mcmc_tools
from scipy.stats import norm
from scipy.stats import gaussian_kde

# id: 個体番号
# y: 生存していた種子数（8個中）
data7a = pandas.read_csv('data7a.csv')
print(data7a.head())
print(data7a.describe())

# モデリング
# 2項ロジスティック回帰でモデリングする。
# 生存率qを導入して、生存する種子数を推定する

Y = data7a['y']
N = len(Y)

stan_data = {
    'y': Y,
    'N': N
}

filename = 'model-exec8-6'
mcmc_result = mcmc_tools.sampling(filename, stan_data,
                                  n_jobs=4,
                                  seed=123)
mcmc_sample = mcmc_result.extract()

# 実測値と予測値のプロット
quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
print(np.percentile(mcmc_result['y_pred'], q=quantile, axis=0))
data_rental_pred = pandas.DataFrame(np.percentile(mcmc_result['y_pred'], q=quantile, axis=0).T, columns=colname)
d = pandas.concat([data7a, data_rental_pred], axis=1)

plt.plot([0,8], [0,8], 'k--', alpha=0.7)
plt.errorbar(d.y, d.p50, yerr=[d.p50 - d.p10, d.p90 - d.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3)
ax = plt.axes()
ax.set_aspect('equal')
plt.legend()
plt.xlabel('Observed', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.show()
plt.close()
# これみるとまあまあ良い性能


