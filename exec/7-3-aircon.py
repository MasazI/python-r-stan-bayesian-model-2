import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm

# 回帰分析の各種テクニックを学んでいく

# 7.3 非線形

# data-aircon
# X: 屋外の平均気温℃
# Y: エアコンの消費電力kWh
data_aircon = pandas.read_csv('data-aircon.txt')
print(data_aircon.head())
print(data_aircon.describe())

# 散布図の表示
# XとYが非線形の関係であることを観察できる
plt.scatter(data_aircon["X"],data_aircon["Y"],c='b')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.show()
plt.close()

# モデリング
# ここでは、快適と感じる温度x_0が存在すると仮定し、消費電力がx_0からの二乗和誤差のような値を平均とした
# 正規分布に従うとする。
Y = data_aircon['Y']
X = data_aircon['X']
N = len(Y)

X_new = np.arange(-10, 35, 1)
N_new = len(X_new)

stan_data = {
    'Y': Y,
    'X': X,
    'N': N,
    'N_new': N_new,
    'X_new': X_new
}

# コンパイル
filename = '../model/model7-3-aircon'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
mcmc_sample = mcmc_result.extract()

# 予測分布
df = pandas.DataFrame(mcmc_sample['Y_new'])
df.columns = X_new
print(df)
qua = [0.025, 0.25, 0.50, 0.75, 0.975]
d_est = pandas.DataFrame()
for i in np.arange(len(df.columns)):
    for qu in qua:
        d_est[qu] = df.quantile(qu)
x = d_est.index
y1 = d_est[0.025].values
y2 = d_est[0.25].values
y3 = d_est[0.5].values
y4 = d_est[0.75].values
y5 = d_est[0.975].values
plt.fill_between(x,y1,y5,facecolor='blue',alpha=0.1)
plt.fill_between(x,y2,y4,facecolor='blue',alpha=0.5)
plt.plot(x,y3,'k-')
plt.scatter(data_aircon["X"],data_aircon["Y"],c='b')
plt.show()
plt.close()

# 予測値と実測値のプロット
quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
print(np.percentile(mcmc_sample['Y_pred'], q=quantile, axis=0))
data_rental_pred = pandas.DataFrame(np.percentile(mcmc_sample['Y_pred'], q=quantile, axis=0).T, columns=colname)
d = pandas.concat([data_aircon, data_rental_pred], axis=1)

plt.plot([-10,35], [-10,35], 'k--', alpha=0.7)
plt.errorbar(d.Y, d.p50, yerr=[d.p50 - d.p10, d.p90 - d.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3)
ax = plt.axes()
ax.set_aspect('equal')
plt.legend()
plt.xlabel('Observed', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.show()
plt.close()
# このデータについて、ほとんどすべての80%信頼区間がy=x上に位置しており、
# 十分に説明変数で応答変数を予測できると考えることができる。