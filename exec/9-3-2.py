# 年齢と年収の関係の問題の説明変数が4個に増えた場合を考える。
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.gridspec import GridSpec
import mcmc_tools
from scipy.stats import norm
import time

# data-attendance-5
# 1行が1名のデータ
# A: アルバイトの好き嫌い
# Score: テストスコア
# X3: 不明
# X4: 不明
# Y: 年収
attendance_5 = pandas.read_csv('data-attendance-5.txt')
print(attendance_5.head())
print(attendance_5.describe())

# 説明変数が多いときに、説明変数それぞれを配列にしてStanに渡すと煩雑になり面倒なので、
# 行列にしてデータを渡す方法がよく使われる

Y = attendance_5['Y']

# 切片と説明変数のDataFrame
# もとのデータからオブジェクトをコピーしておく。
X = attendance_5[['A', 'Score', 'X3', 'X4']].copy()
# Scoreのスケーリング
f_stand = lambda x: x/200
X['Score'] = X['Score'].map(f_stand)
X.insert(0, 'b', 1)

N = len(Y)
D = len(X.columns)

stan_data = {
    'N': N,
    'D': D,
    'X': X,
    'Y': Y
}

# コンパイル
filename = '../model/model9-3-2'
start_1 = time.time()
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
elapsed_time_1 = time.time() - start_1
print("elapsed_time:{0}".format(elapsed_time_1) + "[sec]")

# 実測値と予測値のプロット
# 5-1（model5-3.stan）のプロットよりも、予測と実測値のズレが小さくなっている。（y=x に近づいている）
quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
print(np.percentile(mcmc_result['y_pred'], q=quantile, axis=0))
attendance_5_pred = pandas.DataFrame(np.percentile(mcmc_result['y_pred'], q=quantile, axis=0).T, columns=colname)
d = pandas.concat([attendance_5, attendance_5_pred], axis=1)
d0 = d.query('A==0')
d1 = d.query('A==1')
plt.plot([0,0.5], [0,0.5], 'k--', alpha=0.7)
plt.errorbar(d0.Y, d0.p50, yerr=[d0.p50 - d0.p10, d0.p90 - d0.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3)
plt.errorbar(d1.Y, d1.p50, yerr=[d1.p50 - d1.p10, d1.p90 - d1.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='^', mfc='green', capsize=3)
ax = plt.axes()
ax.set_aspect('equal')
plt.xlabel('Observed', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.show()