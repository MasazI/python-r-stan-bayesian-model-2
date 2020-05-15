# 多項ロジスティック回帰（識別可能性に関する例）
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

Y = category['Y']
X = category[['Age', 'Sex', 'Income']].copy()

f_age_stand = lambda x: x/100
f_income_stand = lambda x: x/1000

X['Age'] = X['Age'].map(f_age_stand)
X['Income'] = X['Income'].map(f_income_stand)

print(X)

X.insert(0, 'b', 1)

print(X)

K = Y.nunique()
N = len(Y)
D = len(X.columns)


stan_data = {
    'N': N,
    'K': K,
    'D': D,
    'X': X,
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
filename = '../model/model10-14-matrix'
start = time.time()
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


