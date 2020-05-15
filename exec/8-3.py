# 非線形モデルの階層モデルを構築する
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.gridspec import GridSpec
import mcmc_tools
from scipy.stats import norm
import time

# 前回は非線形の非階層モデルを構築した。
# 今回は、各グループごとに非線形のモデルに従う階層モデリングを行う。
# つまりグループ間の差異を全体からの誤差だと考える。
# 誤差というのは、正規分布に従うと仮定する。

# data-conc
# PersonID: 1~16
# TimeN: 投与からのN hour経過後の血中濃度
data_conc2 = pandas.read_csv('data-conc-2.txt')
print(data_conc2.head())
print(data_conc2.describe())

# データの図示
# PersonIDごとにデータの分布を見てみる
data_conc2.T.plot()
plt.show()
plt.close()

# 解析の目的
# それぞれの患者に対する効果的な薬の投与タイミングを知りたいとする
# また患者差がどの程度あるかも定量的に判定したいとする
# 患者感の差についてデータを観察するため、各時点における血中濃度のヒストグラムを描いてみる
data_conc2.hist()
plt.show()
# データを観察すると、正規分布かどうかは判断できないが、山型の分布になりそうであることは読み取れる。

# モデリング
# 各Personごとにパラメータをもつのみ、つまり階層化していない場合には、全体の傾向を取り入れることができないため、
# 各患者の切片と線形結合パラメータが特定の平均と分散をもつ聖域分布から観測されるものとする。
# また、患者ごとに予測分布を描きたいとする
Y = data_conc2.loc[:, 'Time1':'Time24']
Time = (1, 2, 4, 8, 12, 24)
N = data_conc2.index.size
T = 6

T_new = 60
Time_new = np.linspace(0, 24, T_new)

stan_data = {
    'Y': Y,
    'Time': Time,
    'N': N,
    'T': T,
    'T_new': T_new,
    'Time_new': Time_new
}

print(Y)

# コンパイル
filename = '../model/model8-3-log'
start = time.time()

mcmc_result = mcmc_tools.sampling(filename, stan_data, adapt_delta=0.99, n_jobs=4, warmup=2000, iter=4000, seed=1234579)

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

mcmc_sample = mcmc_result.extract()


# モデリングをそのままStanにすると非線形関数を左辺にもってくることになってwarningが出る。
# それだけでなく、サンプリング不能になってしまう。
# そのため、Stanで計算するときは、変数変換項で、非線形関数の変換は終わらせて、サンプリングの左辺には
# 非線形関数の結果をもってこないようにすることが必要である。
# そのためmodel8-3を書き換えて、model8-3-logのように記述する。
# このwarningの別の解決方法にはlognormal関数を使うという手がある。
# model8-3ではこの方法を使ってwarningを回避しているが、サンプリングがうまくいかない、、、

# 予測分布
probs = (2.5, 50, 97.5)
qua = np.transpose(np.percentile(mcmc_sample['y_new'], (2.5, 50, 97.5), axis=0), axes=(1, 2, 0))
d_est = pandas.DataFrame(qua.reshape((-1, 3)), columns=['p{}'.format(p) for p in probs])
d_est['PersonID'] = np.repeat(np.arange(N)+1, T_new)
d_est['Time'] = np.tile(Time_new, N)

print(d_est)
Time_tbl = pandas.Series(Time, index=['Time{}'.format(t) for t in Time])
d = pandas.melt(data_conc2, id_vars='PersonID', var_name='Time', value_name='Y')
d['Time'] = Time_tbl[d['Time']].values

_, axes = plt.subplots(4, 4, figsize=figaspect(7/8)*1.5)
for (row, col), ax in np.ndenumerate(axes):
    person = row * 4+ col + 1
    ax.fill_between('Time', 'p2.5', 'p97.5', data=d_est.query('PersonID==@person'), color='k', alpha=1/5)
    ax.plot('Time', 'p50', data=d_est.query('PersonID==@person'), color='k')
    ax.scatter('Time', 'Y', data=d.query('PersonID==@person'), color='k')
    if row < 3:
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        plt.setp(ax, xlabel='Time (hour)')
    if col > 0:
        plt.setp(ax.get_yticklabels(), visible=False)
    else:
        plt.setp(ax, ylabel='Y')
    plt.setp(ax, title=person, xticks=Time, xlim=(0, 24), yticks=np.arange(0, 40, 10), ylim=(-3, 37))
plt.tight_layout()
plt.show()


