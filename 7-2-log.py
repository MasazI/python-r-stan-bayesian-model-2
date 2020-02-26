import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm

# 回帰分析の各種テクニックを学んでいく

# 7.2 対数
# 対数は、変数変換によって解釈がしやすく有用なことが多い。
data_rental = pandas.read_csv('data-rental.txt')
print(data_rental.head())
print(data_rental.describe())

data_rental_log10 = np.log10(data_rental)

# データの図示
# sns.scatterplot(
#     x='Area',
#     y='Y',
#     data=data_rental_log10
# )
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# plt.close()
plt.scatter(data_rental["Area"],data_rental["Y"],c='b')
plt.xlabel('Area', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.xticks([1,2,5,10,20,50,100])
plt.yticks([200,500,1000,2000])
plt.xlim((9, 120))
plt.show()

# AreaをもとにYを推定するモデリング
# Yは正規分布を仮定する Area をもとに平均を推定し、分散をノイズとして扱う。
# ここでは、説明変数、応答変数ともに底10のlogをとって分析を試す。
Y = np.log10(data_rental['Y'])
Area = np.log10(data_rental['Area'])
N = len(Y)
Area_new = np.log10(np.arange(10, 120, 1))
N_new = len(Area_new)

stan_data = {
    'Y': Y,
    'Area': Area,
    'N': N,
    'Area_new': Area_new,
    'N_new': N_new
}

# コンパイル
filename = 'model7-2-log'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
mcmc_sample = mcmc_result.extract()

# 予測分布のプロット
# データ入力時にlog10をとっているので、真数に戻すときは指数をとる
df = np.power(10, pandas.DataFrame(mcmc_sample['Y_new']))
df.columns = np.power(10, Area_new)
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
plt.scatter(data_rental["Area"],data_rental["Y"],c='b')
plt.xticks([1,2,5,10,20,50,100])
plt.yticks([200,500,1000,2000])
plt.xlim((9, 120))
plt.show()
plt.close()

# 予測値と実績値のプロット
quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
print(np.percentile(np.power(10,mcmc_sample['Y_pred']), q=quantile, axis=0))
data_rental_pred = pandas.DataFrame(np.percentile(np.power(10,mcmc_sample['Y_pred']), q=quantile, axis=0).T, columns=colname)
d = pandas.concat([data_rental, data_rental_pred], axis=1)

plt.plot([0,1800], [0,1800], 'k--', alpha=0.7)
plt.errorbar(d.Y, d.p50, yerr=[d.p50 - d.p10, d.p90 - d.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3)
ax = plt.axes()
ax.set_aspect('equal')
plt.legend()
plt.xlabel('Observed', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.show()
plt.close()

# 分散の確認
# 実測値と中央値の差分でMAP推定、カーネル密度推定を行う
data_rental_mu = pandas.DataFrame(np.percentile(mcmc_result['mu'], q=quantile, axis=0).T, columns=colname)
d_mu = pandas.concat([data_rental, data_rental_mu], axis=1)
def ip_diff(d_mu):
    return np.log10(d_mu.loc['Y']) - d_mu.loc['p50']
d_mu['ip'] = d_mu.apply(ip_diff, axis=1)
# MAP推定、カーネル密度推定seabornで実施
# 内部はscipyのnorm
sns.distplot(d_mu['ip'], bins=13, color='#123456', label='data',
             kde_kws={'label': 'kde', 'color': 'k'},
             fit=norm, fit_kws={'label': 'norm', 'color': 'red'},
             rug=False)
plt.legend()
plt.show()
plt.close()
# 7-1.pyで実施した分散の分布と違い、ノイズの多峰性は回避できている。
# 対数をとると、パラメータの誤差は応答変数が大きくなるほど価値をなくしていくためである。