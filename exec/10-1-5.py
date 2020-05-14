# ウサギとカメ（識別可能性に関する例）
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import time

# usagitokame
# Lower: 1カメ、2ウサギ
# Winner: 1カメ、2ウサギ
usagitokame = pandas.read_csv('data-usagitokame.txt')
print(usagitokame.head())
print(usagitokame.describe())

# モデリング
# ここでは、ウサギとカメの強さをモデリングしてみる。
# それぞれの強さは、各動物の平均と標準偏差からサンプリングされると考える。
# データが少ないため、標準偏差=ばらつき=勝負ごとのムラ をウサギとカメの共通とする。

# ここでは勝ち負け予想に強さ（Performance）を導入している。
# 少し考え方に慣れるまで繰り返しサンプリングを頭の中でシミレーションしてみたほうが良い。
# performanceは1<2であるので、これはusagitokameのDataFrameをそのままロードしたものと考えられる。
# ここで、performance1が1である場合、データはmu1からサンプリングされる。performance2が1である場合も、
# データはmu1からサンプリングされる。mu1からサンプリングされるデータは勝つことも負けることもあるが、
# usagitokameのDataFrameで示された結果、performance1<performance2を制約として、サンプリングしていく。
# ということ。これは知らないとなかなか思いつかない。

# また、パフォーマンスはウサギとカメの相対的な数値なので、どちらかが決まらないと並行移動できてしますため、
# 識別不可能になる。そのため、mu[1] つまりカメを0に設定する。
# さらに、平行移動だけでなく、拡大縮小に関して制約を設けないと、まだ識別不可能である。
# そのため、正規分布のsigmalを1に固定する。

N = len(usagitokame)
G = len(usagitokame.columns)

stan_data = {
    'N': N,
    'G': G,
    'LW': usagitokame[['Loser', 'Winner']]
}

# モデリング
filename = 'model10-1-5'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
mcmc_sample = mcmc_result.extract()

# この結果を見ると、カメの強さが0のとき、ウサギの強さ（b）は1.62倍であることがわかる。
# 95%ベイズ信頼区間は（0.89〜2.47）となり、ウサギの強さの幅も推定できる。
