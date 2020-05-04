# target記法を使った方法に書き直す方法
# model9-2-2.stan

import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm
import random

# 年収ファイル2
# X: 年齢、実際からマイナス23（新卒の年齢を23とし、所属年数として扱うため）
# Y: 年収
# KID: 勤務している会社のID(1~4)大手4社

salary2 = pandas.read_csv('data-salary-2.txt')
print(salary2.head())
print(salary2.describe())

# モデリング
# Yを推定するためのパラメータaとbが、全体平均を平均とした正規分布
Y = salary2['Y']
X = salary2['X']
N = len(Y)
N_group = 4
KID = salary2['KID']

stan_data = {
    'Y': Y,
    'X': X,
    'N': N,
    'N_group': N_group,
    'KID': KID
}

# コンパイル
filename = 'model11-exec-1'
# ~からtarget記法に変更
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()
# モデルは等価なので、性能的な違いは有意には出ないが、パラメータが少なくなった分、
# サンプリングにかかる時間が短くなる。
# しかし、モデルによっては逆に時間がかかることもあるらしいので注意。

a_hie = mcmc_sample['a']
pd_hie = pandas.DataFrame(a_hie)
pd_hie.plot.box()
plt.show()
