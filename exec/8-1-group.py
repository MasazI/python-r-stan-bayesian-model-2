# 8-1.pyではグループを考慮せずにパラメータを推定したが、
# ここではグループごとに異なるパラメータが存在すると仮定したモデルを試す。
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.markers import MarkerStyle
import mcmc_tools
from scipy.stats import norm

# 年収ファイル2
# X: 年齢、実際からマイナス23（新卒の年齢を23とし、所属年数として扱うため）
# Y: 年収
# KID: 勤務している会社のID(1~4)大手4社

salary2 = pandas.read_csv('data-salary-2.txt')
print(salary2.head())
print(salary2.describe())

# KIDごとに異なる線形パラメータをもつと仮定して推論を行う。
Y = salary2['Y']
X = salary2['X']
KID = salary2['KID']

N = len(Y)
N_group = len(np.unique(KID.values))

stan_data = {
    'X': X,
    'Y': Y,
    'KID': KID,
    'N': N,
    'N_group': N_group
}

# コンパイル
filename = '../model/model8-1-group'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()

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

# グラフから、今回は各企業ごとに予測と実測値に関しては十分にフィットしていると言えそうである。
# しかしこのモデルは、各社のデータをただ独立に使用しただけであり、全体の様相を推論に反映することができていない。

# 練習問題8(1)
# 会社ごとの年収のベイズ予測区間
# ※ 書籍ではmodel8-2.stanがここではmodel8-1-group.stan
# 予測区間を描く場合は、numpyで正規分布をサンプリングする。
# それがすなわち、ノイズを含んだ予測区間。

N_mcmc = mcmc_sample['lp__'].size

# 勤続年数
X_new = np.arange(salary2['X'].max()+1)
N_X = X_new.size

# 会社ごとにデータをサンプリング
K = salary2['KID'].max()
d2 = pandas.DataFrame([])
for k in range(K):
    # 会社ごとのaを取得
    loc_a = mcmc_sample['a'][:, k].reshape((-1, 1))
    # 会社ごとのbを取得して勤続年数と線形結合
    loc_b = np.outer(mcmc_sample['b'][:, k], X_new)
    loc = loc_a + loc_b
    # 分散をlogと同じshapeにブロードキャスト(分散はサンプルごとでKIDに分かれていないので各社で同じ分散を割り当てる)
    scale = np.broadcast_to(mcmc_sample['sigma'].reshape((-1, 1)), loc.shape)
    # データのサンプリング
    print(loc.shape)
    print(scale.shape)
    y_base_mcmc2 = np.random.normal(loc, scale)
    # 予測区間を含めてデータフレーム化
    d2 = d2.append(pandas.DataFrame(np.hstack([X_new.reshape((-1, 1)), np.percentile(y_base_mcmc2, (2.5, 50, 97.5), axis=0).T, np.full_like(X_new, k+1).reshape((-1, 1))]), columns=('X', 'p2.5', 'p50', 'p97.5', 'KID')), ignore_index=True)

print(d2)

# 2x2のaxを準備
_, axes = plt.subplots(2, 2, figsize=figaspect(2/2), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    kid = i + 1
    q = 'KID==@kid'
    ax.fill_between('X', 'p2.5', 'p97.5', data=d2.query(q), color='k', alpha=1/5)
    ax.plot('X', 'p50', data=d2.query(q), color='k', alpha=0.8, label='')
    ax.scatter('X', 'Y', data=salary2.query(q), c='k', marker=MarkerStyle.filled_markers[i], label=kid)
    ax.legend(title='KID')
    plt.setp(ax, title=kid)
plt.tight_layout()
plt.show()