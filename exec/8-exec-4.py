# 練習問題8(4)、(5)

import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.gridspec import GridSpec
import mcmc_tools
from scipy.stats import norm
from scipy.stats import gaussian_kde

# 学生ごとのデータ
# PersonID: 学生ID
# A: アルバイトが 0嫌い、1好き
# Score: 学力テストのスコア
data_attendance_41 = pandas.read_csv('data-attendance-4-1.txt')
print(data_attendance_41.head())
print(data_attendance_41.describe())

# 出席ごとのデータ
# PersonID: 学生ID
# CourseID: 出席授業のコースID
# Weather: 天気（A晴れ、B曇り、C雨）
# Y: 出席 1した、0していない
data_attendance_42 = pandas.read_csv('data-attendance-4-2.txt')
print(data_attendance_42.head())
print(data_attendance_42.describe())

# 練習問題8(4)
# この問題はサンプリングでなく、単純にデータを観察する
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
data_attendance_42_pid = data_attendance_42.groupby('PersonID')
print(data_attendance_42_pid.groups)
d_person = data_attendance_42_pid['Y'].mean()
sns.distplot(d_person, ax=ax1)

ax2 = fig.add_subplot(1,2,2)
data_attendance_42_cid = data_attendance_42.groupby('CourseID')
sns.distplot(data_attendance_42_cid['Y'].mean(), ax=ax2)
plt.show()

# Scoreのの変換
data_attendance_41['Score'] = data_attendance_41['Score']/200

# 天気データの値を変換
weather_effec = {
    'A': 0,
    'B': 0.2,
    'C': 1
}

data_attendance_42 = data_attendance_42.replace(weather_effec)

# 練習問題8(5)
# 学生差と科目差の線形結合係数が正規分布ではなく、無情報事前分布に従うと仮定する。
# 仮定をなくすとStanではデフォルトで無情報事前分布からのサンプリングとなる。
# normalのサンプリングを削除すればOK
Y = data_attendance_42['Y']
I = len(Y)

# PersonIDごとのA
A = data_attendance_41['A']

# PersonIDごとのScore
Score = data_attendance_41['Score']

# 授業ごとの学生ID
PID = data_attendance_42['PersonID']
N = len(np.unique(PID.values))

# 授業ごとの科目ID
CID = data_attendance_42['CourseID']
C = len(np.unique(CID.values))

# 授業ごとの天気
W = data_attendance_42['Weather']

stan_data = {
    'N': N,
    'C': C,
    'I': I,
    'Y': Y,
    'W': W,
    'PID': PID,
    'CID': CID,
    'Score': Score,
    'A': A

}

# 保存する変数
variables = ['b1', 'b2', 'b3', 'b4', 'bp', 'bc', 'q']

filename = '../model/model-8-exec-4'
mcmc_result = mcmc_tools.sampling(filename, stan_data,
                                  n_jobs=4,
                                  seed=123,
                                  pars=variables)
mcmc_sample = mcmc_result.extract()

# 正規分布のときと同様のグラフを描いて確認

# 各変数の応答変数に対する影響度を確認する。
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.violinplot([mcmc_sample['b1'],
               mcmc_sample['b2'],
               mcmc_sample['b3'],
               mcmc_sample['b4']])
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_xticklabels(['b1', 'b2', 'b3', 'b4'])
ax.set_xlabel('parameters')
ax.set_ylabel('value')
plt.show()
plt.close()

# コースごとの出席率差を表すMAP推定
# mcmcサンプルの数を表す列も追加しておく
param_names = ['mcmc'] + ['bc{}'.format(i+1) for i in range(10)]
N_mcmc = mcmc_sample['lp__'].size
d_est = pandas.DataFrame(np.hstack([np.arange(N_mcmc).reshape((-1, 1)), mcmc_sample['bc']]), columns=param_names)
for i in range(10):
    kde_bc = gaussian_kde(d_est['bc%d' % (i+1)])
    plt.plot(np.linspace(-5, 5, num=100), kde_bc(np.linspace(-5, 5, num=100)), label='bc%d' % (i+1))
plt.xlabel('value')
plt.ylabel('kde density')
plt.legend()
plt.show()
plt.close()