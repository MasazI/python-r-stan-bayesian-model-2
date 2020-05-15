# 打ち切り(Censored)の例
# 打ち切りのあるデータは伝統的な統計学だけで解決策を得ようとしても難しいのだが、
# Stanでは打ち切りの値までの累積分布関数の値を対数尤度に簡単に足せる
# というのもStanではサンプリング中の対数尤度は常にtarget変数に保存されているため、
# targetに累積分布関数の値を足せば良い。
# 打ち切り発生のデータ分の値を足すモデリングをやってみる。
# 打ち切りが発生していないデータのみで通常のサンプリングを実施し、
# 打ち切りが発生しているデータについては、その累積分布関数をtargetに加えるのみとする。

import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm

# ファイルの読み込み
# Y: 血中のタンパク質濃度
protein = pandas.read_csv('data-protein.txt')
# データの先頭5行の確認
print(protein.head())
# データのサマリを確認
print(protein.describe())


# < を含むデータが打ち切り、含まないデータが打ち切りしていないデータである場合
protein_cens = protein.query('Y.str.contains("<")', engine='python')
protein_obs = protein.query('not Y.str.contains("<")', engine='python')

# データに<が含まれるため文字列の認識でDataFrameに格納される。
# floatに変換する
Y_obs = [float(s) for s in protein_obs['Y'].values]
# Y_obs = protein_obs['Y'].astype('float') とpandasの機能を使ってもOK
Y_cens = protein_cens['Y'].values

print(Y_obs)
print(Y_cens)

N_obs =len(Y_obs)
N_cens = len(Y_cens)

# 打ち切り値を取得
L = int(Y_cens[0][1:])

stan_data = {
    'N_obs': N_obs,
    'N_cens': N_cens,
    'Y_obs': Y_obs,
    'L': L
}

# コンパイル
filename = '../model/model7-8'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()

# 打ち切りデータに対する手段として知っておくと良さそう。
# 結果を見ると、平均25.47、分散15.11の正規分布を推定している。