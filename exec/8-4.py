# 非線形のロジスティック回帰モデルを作成する。

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


# Scoreのの変換
data_attendance_41['Score'] = data_attendance_41['Score']/200

# 天気データの値を変換
weather_effec = {
    'A': 0,
    'B': 0.2,
    'C': 1
}

data_attendance_42 = data_attendance_42.replace(weather_effec)

# 解析の目的
# 以前同様の問題を扱ったときには、出席ごとのデータに対して天気が出席確率に影響を及ぼすことを考慮するモデリング
# を行った。
# つまり、切片と天気、スコア、アルバイトの好き嫌いの線形結合をロジスティック関数に噛ませて出席確率として計算し、
# ベルヌーイ分布によって複数回の試行結果を観測していた。
# ここでは、そこに学生ごとの傾向を加える取り組みを行う。
# ここでは大きく分けて、
# 学生ごと、科目ごと、出席授業のコースIDの3種類の要素がある。
# 学生ごと
# アルバイトの好き嫌い、Score、学生差
# 科目ごと
# 科目差
# 授業ごと
# 天気
# これらを最終的に和をとって、ロジスティック関数を通すことで出席確率になると考える。
# 応答変数
# 授業ごとの出席
Y = data_attendance_42['Y']
I = len(Y)

# PersonIDごとのA
A = data_attendance_41['A']
print(A)

# PersonIDごとのScore
Score = data_attendance_41['Score']
print(Score)

# 授業ごとの学生ID
PID = data_attendance_42['PersonID']
print(PID)
N = len(np.unique(PID.values))

# 授業ごとの科目ID
CID = data_attendance_42['CourseID']
print(CID)
C = len(np.unique(CID.values))

# 授業ごとの天気
W = data_attendance_42['Weather']
print(W)

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
print(stan_data)

# 保存する変数
variables = ['b1', 'b2', 'b3', 'b4', 'bp', 'bc', 'sigma_p', 'sigma_c', 'q']
# variables = ['b', 'b_P', 'b_C', 's_P', 's_C', 'q']
# 書籍ではbは配列として保持。配列のほうがスマートで汎用的。

# コンパイル
# stanファイル、最後の確率モデリングでデータごとにせずにベクトルにしてしまうミスがよくある、、、
# データ量の割にサンプリングに時間がかかるのでそれをヒントに気づけるように、、、
# もちろんその前にきづければいいけど
filename = 'model8-4'
mcmc_result = mcmc_tools.sampling(filename, stan_data,
                                  n_jobs=4,
                                  seed=123,
                                  pars=variables)
mcmc_sample = mcmc_result.extract()

# このモデリングは、ある応答変数が、複数のグループの説明変数で構成させる場合の例である。
# この例ではPersonID、CourseIDの2つのグループに依存するデータ存在し、
# 他は授業ごと、つまりデータごと現れるデータ
# だと考え、ロジスティック回帰と組み合わせている。
# 例えば、ある商品の購入確率のモデルを構築したい場合に、顧客に依存する要素、商品に依存する要素、
# 1回ごとの購買に依存する要素（天気、曜日）などに応用が可能である。

# 各変数の応答変数に対する影響度を確認する。
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.violinplot([mcmc_sample['b1'],
               mcmc_sample['b2'],
               mcmc_sample['b3'],
               mcmc_sample['b4'],
               mcmc_sample['sigma_p'],
               mcmc_sample['sigma_c']])
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_xticklabels(['b1', 'b2', 'b3', 'b4', 'sigma_p', 'sigma_c'])
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
# グラフを観察すると、コースごとに出席率への影響度がかなりことなることがわかる。
# MAP推定値（頂点）の差は大きなところで3以上離れているので、オッズはexp(3)=20程度の差が出ていることが読み取れる。
