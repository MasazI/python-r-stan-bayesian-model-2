import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm

# 回帰分析の各種テクニックを学んでいく

# 7.3 非線形

# data-conc
# Time: 投与からの経過時間Time(hour)
# Y: 薬の血中濃度mg/mL
data_conc = pandas.read_csv('data-conc.txt')
print(data_conc.head())
print(data_conc.describe())

# 散布図の表示
# XとYが非線形の関係であることを観察できる
# また、データ点数が非常に少ない
plt.scatter(data_conc["Time"],data_conc["Y"],c='b')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.show()
plt.close()

# このデータのモデリングは、物理法則を用いる。
# 薬の血中濃度yと時間tの関係は、
# 点滴で投与されると考えた場合、
# 一定の流入量Aを仮定する。
# 薬剤は体内で分解される。速度は血中濃度に比例すると考えられているので、
# 血中からの喪失量は -by (b > 0) とかける。
# すると、dy/dt = A - by が単位時間あたりの血中濃度の常備分方程式になる。
# さらにAが b x a だと仮定すると、
# dy/dy = b ( a - y ) となる。
# この常備分方程式を解くと、
# y = a { 1 - Cexp(-bt)} となる。これは定数の置き方に工夫をすることで、曲線フィットにもっていく。
# 説明変数はTime
Y = data_conc['Y']
time = data_conc['Time']
N = len(Y)

time_new = np.linspace(0, 24, 60)
N_new = len(time_new)

print(N_new)

stan_data = {
    'Y': Y,
    'time': time,
    'N': N,
    'time_new': time_new,
    'N_new': N_new
}

# コンパイル
filename = 'model7-3-conc'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()
# model4季では、左辺と右辺の型が異なってもコンパイルが通って計算できてしまうことに注意。
# その場合、パラメータは最後のデータにのみフイットするので、急激にYが増加するようなモデルになる。

# 予測分布
df = pandas.DataFrame(mcmc_sample['Y_new'])
df.columns = time_new
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
plt.scatter(data_conc["Time"],data_conc["Y"],c='b')
# plt.xticks([1,2,5,10,20,50,100])
# plt.yticks([200,500,1000,2000])
#plt.xlim((9, 120))
plt.show()
plt.close()

# 予測値と実績値のプロット
quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
print(np.percentile(mcmc_sample['Y_pred'], q=quantile, axis=0))
data_conc_pred = pandas.DataFrame(np.percentile(mcmc_sample['Y_pred'], q=quantile, axis=0).T, columns=colname)
d = pandas.concat([data_conc, data_conc_pred], axis=1)
plt.plot([0,30], [0,30], 'k--', alpha=0.7)
plt.errorbar(d.Y, d.p50, yerr=[d.p50 - d.p10, d.p90 - d.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3)
ax = plt.axes()
ax.set_aspect('equal')
plt.legend()
plt.xlabel('Observed', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.show()
plt.close()
# ほとんどの実測値が予測値の信頼区間に入っているので推定できていると仮定。