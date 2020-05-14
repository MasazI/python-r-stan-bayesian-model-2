# 交絡(confounding)
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm

# 回帰分析の各種テクニックを学んでいく
# ここでは交絡、モデルの外側に応答変数と説明変数の両方に影響を与える変数が存在する場合。

# data-50m
# Y: 50m走の平均秒速(m/秒)
# Weight: 体重
# Age: 年齢
data_50m = pandas.read_csv('data-50m.txt')
print(data_50m.head())
print(data_50m.describe())

# ペアプロットの表示
sns.pairplot(data_50m, hue="Age", diag_kind='hist')
plt.show()

# ここでのモデリング手法は、パス解析と呼ばれる。
# 因果推論する方法の1つであり、複数の回帰を組み合わせて変数間の因果関係を模索する解析方法である。
# 1つめの回帰は、年齢に対する体重、
# 2つめの回帰は、年齢と体重に対する平均秒速である。
Y = data_50m['Y']
Weight = data_50m['Weight']
Age = data_50m['Age']
N = len(Y)

Age_new = np.arange(7, 12)
N_new = len(Age_new)

stan_data = {
    'Y': Y,
    'Weight': Weight,
    'Age': Age,
    'N': N,
    'Age_new': Age_new,
    'N_new': N_new
}

# コンパイル
filename = 'model7-5'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()
# 1つめの回帰、年齢に対する体重はc2パラメータで表現表現しており、3.96で正の相関。
# 2つめの回帰、年齢と体重に対すパラメータはb2とb3で、b2が0.59で正の相関。
# b3が-0.04で負の相関。
# 理屈と合うパラメータの推定値が得られた。