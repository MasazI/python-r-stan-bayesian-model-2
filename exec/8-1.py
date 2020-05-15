# 階層モデルを学ぶ。
# グループや個人差を扱うための手法の1つであり、最初は階層モデルと単純なモデルを比較して、
# 階層モデルの良い点を観察していく。
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm

# 年収ファイル2
# X: 年齢、実際からマイナス23（新卒の年齢を23とし、所属年数として扱うため）
# Y: 年収
# KID: 勤務している会社のID(1~4)大手4社

salary2 = pandas.read_csv('data-salary-2.txt')
print(salary2.head())
print(salary2.describe())


# 以下の仮定をおく。
# Yは基本年収と正規分布に従うノイズの和である。
# 大手4社は年功序列性だが、新卒の基本年収と年齢に伴う昇給額は会社によってかなり異なる。
# 最後の仮定が、グループ差を考える動機となる。

# ペアプロットの表示
sns.pairplot(salary2, hue="KID", diag_kind='hist')
plt.show()

# 最初は単純な回帰モデルを試す。
# Xを説明変数として、線形モデルを構築、確率分布は正規分布を想定する。
X = salary2['X']
Y = salary2['Y']
N = len(Y)

X_new = np.arange(0, 30)
N_new = len(X_new)

stan_data = {
    'X': X,
    'Y': Y,
    'N': N,
    'N_new': N_new,
    'X_new': X_new
}

# コンパイル
filename = '../model/model8-1'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()

# 予測分布のプロット
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
# plt.scatter(data_rental["Area"],data_rental["Y"],c='b')
plt.show()
plt.close()

# 実測値と予測値のプロット
quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
print(np.percentile(mcmc_result['Y_pred'], q=quantile, axis=0))
salary2_pred = pandas.DataFrame(np.percentile(mcmc_result['Y_pred'], q=quantile, axis=0).T, columns=colname)
d = pandas.concat([salary2, salary2_pred], axis=1)
d0 = d.query('KID==1')
d1 = d.query('KID==2')
d2 = d.query('KID==3')
d3 = d.query('KID==4')

plt.plot([0,800], [0,800], 'k--', alpha=0.7)
plt.errorbar(d0.Y, d0.p50, yerr=[d0.p50 - d0.p10, d0.p90 - d0.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3)
plt.errorbar(d1.Y, d1.p50, yerr=[d1.p50 - d1.p10, d1.p90 - d1.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='^', mfc='green', capsize=3)
plt.errorbar(d2.Y, d2.p50, yerr=[d2.p50 - d2.p10, d2.p90 - d2.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='x', mfc='red', capsize=3)
plt.errorbar(d3.Y, d3.p50, yerr=[d3.p50 - d3.p10, d3.p90 - d3.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='yellow', capsize=3)

ax = plt.axes()
ax.set_aspect('equal')
plt.legend()
plt.xlabel('Observed', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.show()
plt.close()

# グラフから、会社によって金額を大きめに予想しすぎたり、小さめに予想しすぎたりすることが発生していることがわかる。
# また、KID=4については観測と予測値は負の相関を示している。
