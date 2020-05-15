# Stanを扱ううえで役に立つトラブルシューティングを学ぶ
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.gridspec import GridSpec
import mcmc_tools
from scipy.stats import norm
import time


# Stanではint型の整数値でパラメータを定義できない
# 等価なモデル構築については、11章


# 欠損値を扱う
# data-conc-2-NA-wide.txt
# PersonID: 人ID
# Time1,2,8,12,24: 各時間における薬剤の血中濃度
conc_2_na_long = pandas.read_csv('data-conc-2-NA-wide.txt')
print(conc_2_na_long)
print(conc_2_na_long.head())
print(conc_2_na_long.describe())

# このようなNAを含むデータは横長（列固定）ではなく、縦長（ID指定）へ変形するとStan内で測定ごとにループを回しやすくなる

# data-conc-2-NA-long.txt
# PersonID: 人ID
# TimeID: 時間ID
# Y: 薬剤の血中濃度
conc_2_na_long = pandas.read_csv('data-conc-2-NA-long.txt')
print(conc_2_na_long)
print(conc_2_na_long.head())
print(conc_2_na_long.describe())

# 物理計算から、患者ごとに Y=a{1-exp(-bt)} に当てはめるのだが、縦長(long)になると、TimeIDを組み込む必要がある。
# そのため、平均muをTimeIDごとに定義する。
# aは頭打ちになるYの値、bが頭打ちになるまでの時間を決めるパラメータであり、全体の傾向を表す正規分布から生成される。
Y = conc_2_na_long['Y']
I = len(Y)
PersonID = conc_2_na_long['PersonID']
N = PersonID.nunique()

TimeID = conc_2_na_long['TimeID']
T = TimeID.nunique()

Time = np.array([1, 2, 4, 8, 12, 24])

T_new = 60
Time_new = np.linspace(0, 24, T_new)

stan_data = {
    'I': I,
    'N': N,
    'T': T,
    'PersonID': PersonID,
    'TimeID': TimeID,
    'Time': Time,
    'Y': Y,
    'T_new': T_new,
    'Time_new': Time_new
}

# コンパイル
filename = '../model/model9-5'
start = time.time()
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


# 予測分布
probs = (2.5, 50, 97.5)
qua = np.transpose(np.percentile(mcmc_result['y_new'], (2.5, 50, 97.5), axis=0), axes=(1, 2, 0))
d_est = pandas.DataFrame(qua.reshape((-1, 3)), columns=['p{}'.format(p) for p in probs])
d_est['PersonID'] = np.repeat(np.arange(N)+1, T_new)
d_est['Time'] = np.tile(Time_new, N)

print(d_est)
Time_tbl = pandas.Series(Time, index=['Time{}'.format(t) for t in Time])
d = pandas.melt(conc_2_na_long, id_vars='PersonID', var_name='Time', value_name='Y')
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
    plt.setp(ax, title=person, xticks=Time, xlim=(0, 24), yticks=np.arange(0, 60, 10), ylim=(-3, 57))
plt.tight_layout()
plt.show()
# 結果をみると、PersonID3のデータだけ欠損値が3つあるため、信頼区間が大きく描かれることがわかる。
# また、Stanで予測分布を推定しなくても、muの値を使ってPythonを利用して正規分布からデータをサンプリングすれば、同様の結果を得ることができる。


