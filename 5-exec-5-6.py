import numpy as np
import seaborn as sns
import pandas
import mcmc_tools
import matplotlib.pyplot as plt
from scipy.stats import norm

# ファイルの読み込み
# y: 種子数 応答変数
# x: 体サイズ
# f: 処理の違い（CまたはT）
data3a = pandas.read_csv('data3a.csv')
print(data3a.head())
print(data3a.describe())

# 目的
# 体サイズXと処理の違いfがyに及ぼず影響を調べる。
# そのため、xとfからyを推定できるかモデリングする。

# 応答変数は離散的な自然数なのでポアソン分布を利用する。
# ポアソン分布に与えるパラメータはxとfを線形結合して指数をとったものとする。
# カテゴリカル変数であるfは0と1に変換して利用する。
# 1と2でも良いが、バイナリ値である場合には掛け算でパラメータの使用の有無を決定するほうが計算がシンプルにできる。
# 最初はパラメータの数を増やしても良いが、その後削減できるパラメータがないかどうか確認することも大切である。
# また、expとpoissonの組み合わせてうまくパラメータが探索できないとき、
#
f_effec = {
    'C': 0,
    'T': 1
}

data3a_re = data3a.replace(f_effec)
print(data3a_re.head())

y = data3a_re['y']
x = data3a_re['x']
f = data3a_re['f']
N = len(y)

# データ準備
stan_data = {
    'N': N,
    'y': y,
    'x': x,
    'f': f
}

# コンパイル
filename = '5-exec-5-6'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=1983)
mcmc_sample = mcmc_result.extract()

# 予測値と実測値のプロット
quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
print(np.percentile(mcmc_sample['y_pred'], q=quantile, axis=0))
data3a_pred = pandas.DataFrame(np.percentile(mcmc_sample['y_pred'], q=quantile, axis=0).T, columns=colname)
# 実装値と予測値のDataFrameを作成
d = pandas.concat([data3a_re, data3a_pred], axis=1)
d0 = d.query('f==0')
d1 = d.query('f==1')

plt.errorbar(d0.y, d0.p50, yerr=[d0.p50 - d0.p10, d0.p90 - d0.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3, label="f=0")
plt.errorbar(d1.y, d1.p50, yerr=[d1.p50 - d1.p10, d1.p90 - d1.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='^', mfc='green', capsize=3, label="f=1")
ax = plt.axes()
plt.plot([0,15], [0,15], 'k--', alpha=0.7)
ax.set_aspect('equal')
plt.legend()
plt.xlabel('Observed', fontsize=20)
plt.ylabel('Predicted', fontsize=20)
plt.show()