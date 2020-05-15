# 離散値をとるパラメータを使う
# data-coin.txtのサンプルを使って、コインを2枚投げた後とする。

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
#

print(coin.head())
print(coin.describe())

Y = coin['Y']
N = len(Y)

stan_data = {
    'N': N,
    'Y': Y
}

# コンパイル
# A)3
filename = '../model/model11-exec-3'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()
# qの中央値と95%信頼区間
# qの平均は0.04（4%）
# 1 %~13 %が95%信頼区間


filename = '../model/model11-exec-3b'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()
# qの平均は0.13（13%）
# 6%〜33%が95%信頼区間
# p_coinの平均は88%
# 68%~99%が95%信頼区間