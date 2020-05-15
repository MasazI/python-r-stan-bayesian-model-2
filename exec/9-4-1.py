# パラメータに制約を入れるいくつかの方法
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.gridspec import GridSpec
import mcmc_tools
from scipy.stats import norm
import time

# data-dice.txt
# 200回分のサイコロを降った結果の記録
# Face: 出た目
dice = pandas.read_csv('data-dice.txt')
print(dice)
print(dice.head())
print(dice.describe())

# カテゴリカルな問題では、試行ごとに扱うよりも集計をとって多項分布を使ったほうがはるかに高速であるらしい。
Y = dice['Face'].value_counts().sort_index()
K = len(Y)

stan_data = {
    'Y': Y,
    'K': K
}
# コンパイル
filename = '../model/model9-4-1'
start_1 = time.time()
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
elapsed_time_1 = time.time() - start_1
print("elapsed_time:{0}".format(elapsed_time_1) + "[sec]")

# カテゴリカルデータを1試行ずつ行った場合のモデリング（倍以上の計算時間がかかる）
Y_b = dice['Face']
N = len(Y_b)

stan_data_b = {
    'Y': Y_b,
    'N': N,
    'K': K
}
# コンパイル
filename = '../model/model9-4-1-b'
start_1 = time.time()
mcmc_result_b = mcmc_tools.sampling(filename, stan_data_b, n_jobs=4)
elapsed_time_1 = time.time() - start_1
print("elapsed_time:{0}".format(elapsed_time_1) + "[sec]")
