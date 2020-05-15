# 離散値をとるパラメータを使う
# 場合の数を数え上げて、離散パラメータを消去した（周辺化消去）形で対数尤度を表現する
# target記法
# log_sum_exp関数

# コインのパターン
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm
import random

# 喫煙率調査ファイル
# Y: 0非喫煙、1喫煙
coin = pandas.read_csv('data-coin.txt')
print(coin.head())
print(coin.describe())

Y = coin['Y']
N = len(Y)

stan_data = {
    'N': N,
    'Y': Y
}

# コンパイル
filename = '../model/model11-1'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()

#        mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
# q      0.21  3.5e-3   0.09   0.03   0.14   0.21   0.26   0.38    649    1.0
# lp__  -69.7    0.04   0.91 -72.44 -69.86 -69.32 -69.13 -69.08    564    1.0