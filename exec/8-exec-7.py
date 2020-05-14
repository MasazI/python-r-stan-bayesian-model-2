# 練習問題8(7)
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.gridspec import GridSpec
import mcmc_tools
from scipy.stats import norm
from scipy.stats import gaussian_kde

# id: 個体番号
# pot: 植木鉢番号(A~J)
# f: 処理の違い(C or T)
# y: 種子数
d1 = pandas.read_csv('d1.csv')
print(d1.head())
print(d1.describe())

# モデリング
# 線形結合するのは"処理の違い"とし、ほかはノイズとして扱う。
y = d1['y']
N = len(y)

# pandasでIDを変更するときはSeriesが使える
F2int = pandas.Series([0, 1], index=('C', 'T'))
F = F2int[d1['f']]
print(F)

# こちらもIDを変更するのでSeriesを使う
pots = d1['pot'].unique()
N_Pot = pots.size
Pot2int = pandas.Series(np.arange(N_Pot)+1, index=pots)
N2Pot = Pot2int[d1['pot']]
print(N2Pot)

stan_data = {
    'Y': y,
    'N': N,
    'N2P': N2Pot,
    'F': F,
    'N_Pot': N_Pot
}

# モデリング
# この例題では説明変数は"処理の違い"のみになっている。
# その他は個体IDと鉢に依存するノイズを生成して、ポアソン分布のパラメータに追加する形になっている。
filename = 'model-exec8-7'
mcmc_result = mcmc_tools.sampling(filename, stan_data,
                                  n_jobs=4,
                                  seed=123)
mcmc_sample = mcmc_result.extract()

# 個体差はb_I、鉢の差はb_Pに反映されるので、MAP推定値のヒストグラムを描いてみる。
# まずは個体差
param_names = ['mcmc'] + ['b_I{}'.format(i+1) for i in range(100)]
N_mcmc = mcmc_sample['lp__'].size
d_est = pandas.DataFrame(np.hstack([np.arange(N_mcmc).reshape((-1, 1)), mcmc_sample['b_I']]), columns=param_names)
for i in range(100):
    kde_bc = gaussian_kde(d_est['b_I%d' % (i+1)])
    plt.plot(np.linspace(-5, 5, num=100), kde_bc(np.linspace(-5, 5, num=100)), label='b_I%d' % (i+1))
plt.xlabel('value')
plt.ylabel('kde density')
plt.show()
plt.close()

# 次に鉢
param_names = ['mcmc'] + ['b_P{}'.format(i+1) for i in range(10)]
N_mcmc = mcmc_sample['lp__'].size
d_est = pandas.DataFrame(np.hstack([np.arange(N_mcmc).reshape((-1, 1)), mcmc_sample['b_P']]), columns=param_names)
for i in range(10):
    kde_bc = gaussian_kde(d_est['b_P%d' % (i+1)])
    plt.plot(np.linspace(-10, 10, num=100), kde_bc(np.linspace(-10, 10, num=100)), label='b_P%d' % (i+1))
plt.xlabel('value')
plt.ylabel('kde density')
plt.legend()
plt.show()
plt.close()
# 鉢植えごとの差は大きなところで5。オッズ計算でexp(5)くらい差があることになる。
# けっこう違う。