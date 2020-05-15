import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm

# 説明変数にノイズを含む場合
# 説明変数自体が推定値である場合には、そもそも説明変数にノイズが存在していると考えることができる。
# data-salalyの年齢Xがアプリケーションによる推定値だったと仮定した場合。

# ファイルの読み込み
# X: 年齢
# Y: 年収
salary = pandas.read_csv('data-salary.txt')

# データの先頭5行の確認
print(salary.head())

# データのサマリを確認
print(salary.describe())

# データの図示
sns.scatterplot(
    x='X',
    y='Y',
    data=salary
)
plt.show()

# Xに+-2.5程度のノイズを仮定したい場合、標準偏差2.5の正規分布を使うことができる。
Y = salary['Y']
X = salary['X']
N = len(Y)

stan_data = {
    'Y': Y,
    'X': X,
    'N': N
}

# コンパイル
filename = '../model/model7-7'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()

# 説明変数にノイズを含む場合の扱いが、ベイジアンモデリングを使うことで簡単にできる例である。
