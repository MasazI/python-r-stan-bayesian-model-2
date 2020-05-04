# ポアソン分布に従う離散パラメータ
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm
from scipy.stats import binom
import random

# ポアソン分布に従う枚数だけコインを投げた実験結果
# Y: 表が出たコインの枚数
poisson_binomial = pandas.read_csv('data-poisson-binomial.txt')
print(poisson_binomial.head())
print(poisson_binomial.describe())

# 周辺化の積分範囲を決定するため、二項分布の確率を確認する。
b = binom.pmf(range(40),40,0.5)
plt.plot(range(40), b, 'o')
plt.xlabel('number of head', fontsize=15)
plt.ylabel('probability', fontsize=15)
plt.show()
plt.close()
# 40まで計算すると、10枚程度が表になる確率はとても小さいことがわかる。

b = binom.pmf(range(20),20,0.5)
plt.plot(range(20), b, 'o')
plt.xlabel('number of head', fontsize=15)
plt.ylabel('probability', fontsize=15)
plt.show()
# 20までにすると、10枚の確率が17.5%と最も高く、9枚もその周辺で10%以上である。

Y = poisson_binomial['Y']
N = len(Y)

stan_data = {
    'Y': Y,
    'N': N
}


# コンパイル
filename = 'model11-1-2'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()

# これによってlambdaの平均、中央値、95%信頼区間などを得ることができる。
# しかし、この計算方法は周辺化の計算に大きなコストがかかるため、n=40のような小さい値を覗いて現実液には
# 計算が終わらない可能性が高い。
# その場合、周辺化を簡潔にする公式があるかどうか確認して、あれば使うと良い。

# コンパイル
filename_b = 'model11-1-2b'
mcmc_result_b = mcmc_tools.sampling(filename_b, stan_data, n_jobs=4, seed=123)
mcmc_sample_b = mcmc_result_b.extract()
# ポアソン分布の周辺化は非常にシンプルな式で計算が可能。