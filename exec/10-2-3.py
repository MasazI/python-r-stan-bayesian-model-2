# プロ棋士の強さと勝負ムラ
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import time

# data-shogi-player
# 166 人、6231 試合
# Lower: 棋士ID(1〜166)
# Winner: 棋士ID(1〜166)
shogi_player = pandas.read_csv('data-shogi-player.txt')
print(shogi_player.head())
print(shogi_player.describe())

LW = shogi_player[['Loser', 'Winner']]
G = len(LW)

shogi_player_name = pandas.read_table('data-shogi-player-name.txt')
print(shogi_player_name.head())
print(shogi_player_name.describe())

N = shogi_player_name['nid'].nunique()

stan_data = {
    'G': G,
    'N': N,
    'LW': LW
}

# モデリング
filename = '../model/model10-2-3'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
mcmc_sample = mcmc_result.extract()

# ランキングの作成
mu_values = mcmc_sample['mu'].T
mu_median = np.median(mu_values, axis=1)
print(mu_median)

mu_median_df = pandas.DataFrame(mu_median.T, columns=['performance'])
print(mu_median_df)

shogi_player_mu_median = pandas.concat([shogi_player_name, mu_median_df], axis=1)

print(shogi_player_mu_median.nlargest(5, 'performance'))

# 以下のようなランキングが出力される（performanceで数値化される）
#      kid kname  nid  performance
# 46   175  羽生善治   47     1.862576
# 104  235   渡辺明  105     1.600026
# 133  264  豊島将之  134     1.352212
# 77   208  行方尚史   78     1.301943
# 64   195  郷田真隆   65     1.272320
# また、勝負ムラのパラメータを観察すれば、ムラの小さな棋士と大きな棋士をリスト化することもできる。