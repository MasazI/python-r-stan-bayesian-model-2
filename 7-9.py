import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm

# 最後は外れ値を含むデータ
# これは知っていれば非常に簡単な手法。
# 裾の重い分布を使って外れ値の確率分布への影響を減らすことを試みる。

# ファイルの読み込み
# X: なにかのインデックス
# Y: なにかの数値データ
outlier = pandas.read_csv('data-outlier.txt')
# データの先頭5行の確認
print(outlier.head())
# データのサマリを確認
print(outlier.describe())

# データの図示
sns.scatterplot(
    x='X',
    y='Y',
    data=outlier
)
plt.show()
# xが3のあたりで極端なYを記録しており、外れ値と想定したとする。
# コーシー分布を仮定することで、予測分布への外れ値の影響が小さくなっていることを確認する。

X = outlier['X']
Y = outlier['Y']
X_new = np.linspace(0, 11, 100)

N = len(Y)
N_new = len(X_new)

stan_data = {
    'N': N,
    'X': X,
    'Y': Y,
    'N_new': N_new,
    'X_new': X_new
}

# コンパイル
filename = 'model7-9'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()

# 予測分布
df = pandas.DataFrame(mcmc_sample['Y_new'])
df.columns = X_new
print(df)
qua = [0.025, 0.25, 0.50, 0.75, 0.975]
d_est = pandas.DataFrame()
for i in np.arange(len(df.columns)):
    for qu in qua:
        d_est[qu] = df.quantile(qu)

x = d_est.index
y1 = d_est[0.025].values
y2 = d_est[0.25].values
y3 = d_est[0.5].values
y4 = d_est[0.75].values
y5 = d_est[0.975].values

plt.fill_between(x,y1,y5,facecolor='blue',alpha=0.1)
plt.fill_between(x,y2,y4,facecolor='blue',alpha=0.5)
plt.plot(x,y3,'k-')
plt.scatter(outlier["X"],outlier["Y"],c='b')
plt.show()
plt.close()

# 予測分布を確認すると、外れ値による信頼区間への影響が小さいことが確認できる。
# しかしながら、単純な正規分布を仮定した場合とどちらが正解かというと、
# 与えられたデータからだけでは判断できない。
# xが3付近では別の理由で大きな値をとるのかもしれない。
# 今回はxが3付近のデータをノイズと仮定した場合の確率分布を推定したにすぎない。

# 別の方法として、混合正規分布やZIP(Zero Inflated Poisson)分布を使うと良い。
# Zero Inflated Poissonは、ベルヌーイ分布とポアソン分布を混ぜた分布で、分布中に
# A もしくは Bというようなカテゴリカルなクラスが存在し、それぞれごとにポアソン分布で
# 回数を観測しているような場合に用いる。
# これらの分布については11章で再度扱うらしい。