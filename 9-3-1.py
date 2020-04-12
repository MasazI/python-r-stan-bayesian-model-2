# 多変量正規分布はベクトルが従う分布と考えることができる。
# そのためStanで扱う場合はベクトルとmatrixを使って表現する。
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.gridspec import GridSpec
import mcmc_tools
from scipy.stats import norm
import time

# data-mvn
# 1行が1名のデータ
# Y1: 50m走のタイム(sec)
# Y2: 走り幅跳びで飛んだ距離(m)
mvn = pandas.read_csv('data-mvn.txt')
print(mvn.head())
print(mvn.describe())

# 散布図を描いてみる
sns.scatterplot(
    x='Y1',
    y='Y2',
    data=mvn
)
plt.show()
# 足が速い（秒数が少ない）ほうが、幅跳びの距離が大きくなる
# 負の相関が見られる。

# モデリング
# ここでは各人ごとに平均ベクトルと分散共分散行列をもつ多変量正規分布から
# Y1とY2のベクトルデータ（2次元）が観測されるとする。
Y1 = mvn['Y1']
Y2 = mvn['Y2']
N = len(Y1)
D = 2

stan_data = {
    'N': N,
    'D': D,
    'Y': mvn
}

# コンパイル
filename = 'model9-3-1'
start_1 = time.time()
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
elapsed_time_1 = time.time() - start_1

# ベクトル化による高速化
filename = 'model9-3-1-vec'
start_2 = time.time()
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
elapsed_time_2 = time.time() - start_2

print("elapsed_time:{0}".format(elapsed_time_1) + "[sec]")
print("elapsed_time:{0}".format(elapsed_time_2) + "[sec]")

