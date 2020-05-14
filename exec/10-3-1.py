# 再パラメータ化 reparameterization
# 目的: サンプリングの効率化
# データのスケーリングもreparameterizationの一種
# 最初は極端な例、"Nealの漏斗"を見てみる。

import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
import scipy.stats as stats
from scipy.stats import norm
import random


# Nealの漏斗
# データが無く、事前分布がそのまま事後分布になる。
# 対数事後分布がいびつだとうまくサンプリングできない、ということの例なのだが、なかなか理解難しいな。
# 対数事後分布の値によってサンプリングするわけなので、影響があるのは理解できる。
# 対策
# 分布からスケールを切り離す。
stan_data = {}
filename = 'model10-3-1'
mcmc_result_b = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result_b.extract()

# 2次元メッシュの書き方を復習
X, Y = np.mgrid[-5:6, -5:6]
print(X.shape)
print(X)
print(Y.shape)
print(Y)

# print(X.ravel())
pts = np.c_[X.ravel(), Y.ravel()]
# print(pts[:,0])

data = X+2*Y
# print(data.shape)
# print(data)

plt.contourf(X, Y, data)
plt.show()

# サンプリングの結果を使って2次元メッシュを作成する
xx, yy = np.mgrid[-5:5:30j, -5:5:30j]
x = xx.ravel()
y = yy.ravel()
# メッシュの座標をすべて軸ごとに1つの配列にいれたものを用意
print(x)
# 座標ごとに対数事後確率を計算
lp = np.log(stats.norm.pdf(yy, loc=0, scale=3)) + np.log(stats.norm.pdf(xx, loc=0, scale=np.exp(yy/2)))
lp[lp < -15] = -15
plt.contourf(xx, yy, lp, vmin=-15, vmax=0)
plt.scatter(mcmc_sample['r'][:, 0], mcmc_sample['a'], s=1, c='k')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
plt.close()

# 再パラメータ化したもの
# これは、もとのパラメータを独立な正規分布からサンプリングして、サンプリング時に計算される
# 対数事後分布の大きさによってサンプリングが偏らないようにできるというもの
filename = 'model10-3-1-b'
mcmc_result_b = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample_b = mcmc_result_b.extract()

lp = np.log(stats.norm.pdf(yy, loc=0, scale=3)) + np.log(stats.norm.pdf(xx, loc=0, scale=np.exp(yy/2)))
lp[lp < -15] = -15
plt.contourf(xx, yy, lp, vmin=-15, vmax=0)
plt.scatter(mcmc_sample_b['r'][:, 0], mcmc_sample_b['a'], s=1, c='k')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
