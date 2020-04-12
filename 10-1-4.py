# 多項ロジスティック回帰
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import time

# 商品購買
# Age: 年齢
# Sex: 性別
# Income: 収入
# Y: 購買商品ID
category = pandas.read_csv('data-category.txt')
print(category.head())
print(category.describe())

# 変数のスケーリングはかなり重要
# スケーリングしないと収束しないパターン。
# ベイズモデリングでは、サンプリングの効率を考えて、できるだけデータを原点周辺にもってくることが大切。
Y = category['Y']
Age = category['Age']/100
Sex = category['Sex']
Income = category['Income']/1000

K = Y.nunique()
N = len(Y)

stan_data = {
    'N': N,
    'K': K,
    'Age': Age,
    'Sex': Sex,
    'Income': Income,
    'Y': Y
}

# モデリングの工夫
# クラスKのカテゴリカル分布を使う場合、
# 次元がKのベクトルをデータい掛け合わせて線形結合をつくり（同様に次元Kのベクトル）、
# softmax関数への入力とするが、直感的な説明だと各ベクトルの次元の強さだと捉えることができる。
# しかしこのベクトルは空間内で並行移動（原点を移動）させても同じ作用をもつので、
# 識別が不可能になる。
# そのため、あるカテゴリーを選択する強さを定数で固定する。定数ならなんでも良いが、
# 0に固定するのがシンプルでわかりやすい。
# 固定する前は事後確率が安定せずサンプリングがうまく行かないが、固定すると進む。

# コンパイル
filename = 'model10-1-4'
start = time.time()
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


