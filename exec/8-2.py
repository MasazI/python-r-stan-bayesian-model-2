# 複数階層をもつモデリング
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


# 年収ファイル3
# X: 年齢、実際からマイナス23（新卒の年齢を23とし、所属年数として扱うため）
# Y: 年収
# KID: 勤務している会社のID(1~30)30社
# GID: 業界のID(1~3)
salary3 = pandas.read_csv('data-salary-3.txt')
print(salary3.head())
print(salary3.describe())

# データの図示
sns.scatterplot(
    x='X',
    y='Y',
    hue='GID',
    data=salary3
)
plt.show()

# IDが増えてきたら各データの分布を確認するのも良い方法である。
# 各IDごとに単回帰してみる。

# 単回帰 sklearn
salary_g1 = salary3.query('GID==1')
salary_g2 = salary3.query('GID==2')
salary_g3 = salary3.query('GID==3')

# 単回帰 Seaborn
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey='all', sharex='all')
sns.regplot("X","Y",salary_g1, ax=ax1)
sns.regplot("X","Y",salary_g2, ax=ax2)
sns.regplot("X","Y",salary_g3, ax=ax3)
plt.tight_layout()
plt.show()
plt.close()

x = salary_g1[['X']]
y = salary_g1[['Y']]
model_lr = LinearRegression()
model_lr.fit(x, y)

# モデリング
# 会社ごとに切片と線形結合係数を想定し、それらが全体平均と業界平均から構成されていると考える。
# 業界平均は全体平均を平均とした正規分布から観測されると考える。
# 各会社ごとの切片と線形結合係数は、業界平均を平均とした正規分布から観測されると考える。

Y = salary3['Y']
X = salary3['X']
N = len(Y)
N_group = 30
KID = salary3['KID']
N_industry = 3
GID = salary3['GID']

# 会社IDから業界IDを取得するデータを取得する配列を作成
K2G = salary3.drop_duplicates(subset=['KID', 'GID'])['GID']

stan_data = {
    'Y': Y,
    'X': X,
    'N': N,
    'N_group': N_group,
    'N_industry': N_industry,
    'KID': KID,
    'K2G': K2G,
    'GID': GID
}

# コンパイル
filename = '../model/model8-2'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()

# このモデリングでは、業界ごとに変化するのは平均のみと仮定しているが、
# 分散も変化すると仮定するモデリングを行う。
#
filename = '../model/model8-2-sd'
mcmc_result_sd = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample_sd = mcmc_result.extract()
# もしモデルが収束しない場合、複雑さに比べてデータが少ないことが原因が考えられる。

# 練習問題8(3)
# 会社ごとの年収のベイズ予測区間
# ※ 書籍ではmodel8-5.stanがここではmodel8-2.stan
# a業界平均[g]とb業界平均[g]の事後分布を描く
N_mcmc = mcmc_sample['lp__'].size
G = salary3['GID'].max()

print(N_mcmc)
print(G)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.violinplot(mcmc_sample['a1'])
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['a1_1', 'a1_2', 'a1_3'])
ax.set_xlabel('parameters')
ax.set_ylabel('value')
plt.show()
plt.close()

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.violinplot(mcmc_sample['b1'])
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(['b1_1', 'b1_2', 'b1_3'])
ax2.set_xlabel('parameters')
ax2.set_ylabel('value')
plt.show()
plt.close()
