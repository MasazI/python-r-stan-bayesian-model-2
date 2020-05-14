# 階層モデルを似変量正規分布で拡張する例に弱情報事前分布を組み込む
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

# このデータに以下の背景知識を弱情報事前分布として組み込む。
Y = salary2['Y']
N = len(Y)

X = salary2['X']
KID = salary2['KID']
N_group = salary2['KID'].nunique()

stan_data = {
    'Y': Y,
    'N': N,
    'X': X,
    'KID': KID,
    'N_group': N_group
}

# # コンパイル
# filename = 'model10-2-4'
# mcmc_result_b = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
# mcmc_sample = mcmc_result_b.extract()
#
#
# # コンパイル
# # パラメータを再構成して制約を与えたパターン
# # ただし、この方法はパラメータが増えた場合に制約を設定しにくい。
# filename = 'model10-2-4-b'
# mcmc_result_b = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
# mcmc_sample_b = mcmc_result_b.extract()

# コンパイル
# LKJ相関分布を使う方法
# 知らないと適用が難しいが、変数間の相関を事前分布として設定するのによく使われるとのこと。
filename = 'model10-2-4-c'
stan_data_c = {
    'Y': Y,
    'N': N,
    'X': X,
    'KID': KID,
    'N_group': N_group,
    'Nu': 2
}
mcmc_result_c = mcmc_tools.sampling(filename, stan_data_c, n_jobs=4, seed=123)
mcmc_sample_c = mcmc_result_c.extract()
# このモデリングでは、相関行列と分散共分散行列を求める方法も示している。
# a と b の関係を得ることができる。