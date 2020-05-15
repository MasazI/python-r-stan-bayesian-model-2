import numpy as np
import seaborn as sns
import pandas
import mcmc_tools
import matplotlib.pyplot as plt
from scipy.stats import norm

# ファイルの読み込み
# y: 生存していた種子数 応答変数
# N: 調査した種子数
# x: 体サイズ
# f: 処理の違い（CまたはT）
data4a = pandas.read_csv('data4a.csv')
print(data4a.head())
print(data4a.describe())

# ここでは、2項ロジスティック回帰を利用して推定を試みる。
# 種子の生存率を確立で表すため、ロジスティック関数を利用する。
# 確率はxとfに依存するものとする。
# その確率と、調査した種子数を説明変数として、生存していた種子数をモデリングする。
f_effec = {
    'C': 0,
    'T': 1
}

data4a_re = data4a.replace(f_effec)
print(data4a_re.head())

y = data4a_re['y']
x = data4a_re['x']
f = data4a_re['f']
N = data4a_re['N']
I = len(y)

stan_data = {
    'I': I,
    'y': y,
    'N': N,
    'x': x,
    'f': f
}

# コンパイル
filename = '../model/5-exec-5-7'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=1983)
mcmc_sample = mcmc_result.extract()


# 予測値と実測値のプロット
quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
print(np.percentile(mcmc_sample['y_pred'], q=quantile, axis=0))
data4a_pred = pandas.DataFrame(np.percentile(mcmc_sample['y_pred'], q=quantile, axis=0).T, columns=colname)
# 実装値と予測値のDataFrameを作成
d = pandas.concat([data4a_re, data4a_pred], axis=1)
d0 = d.query('f==0')
d1 = d.query('f==1')

plt.errorbar(d0.y, d0.p50, yerr=[d0.p50 - d0.p10, d0.p90 - d0.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3, label="f=0")
plt.errorbar(d1.y, d1.p50, yerr=[d1.p50 - d1.p10, d1.p90 - d1.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='^', mfc='green', capsize=3, label="f=1")
ax = plt.axes()
plt.plot([0,10], [0,10], 'k--', alpha=0.7)
ax.set_aspect('equal')
plt.legend()
plt.xlabel('Observed', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.show()

# この結果から、練習問題5(6)よりも5(7)の設定のほうが良い推定を得られることがわかる。
# これはデータとデータをつなぐ仕組みを考えることにつながる。説明変数はほとんど変わらないのに、
# 種子が生き残る確率、とと検査した個体数を導入することによって、
# 推論性能が上昇していることがわかる。
# これが、スライダークランク機構を例に題して説明される実例である。