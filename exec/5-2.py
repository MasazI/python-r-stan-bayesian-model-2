import numpy as np
import seaborn as sns
import pandas
import mcmc_tools
import matplotlib.pyplot as plt
from scipy.stats import norm

# ファイルの読み込み
# PersonID: 学生のID
# M: 3ヶ月間ヶ月間における履修登録した科目の総授業回数
# Y: Mのうち出席した授業回数
attendance_2 = pandas.read_csv('data-attendance-2.txt')
print(attendance_2.head())
print(attendance_2.describe())

# ratio = Y/M （出席率）を追加
attendance_2['ratio'] = attendance_2.apply(lambda x: x['Y']/x['M'], axis=1)
# 確認
print(attendance_2.head())

# データ観察作業として散布部行列の確認のため、ペアプロットの表示
sns.pairplot(attendance_2, hue="A", diag_kind='hist', vars=['A', 'Score', 'M', 'Y', 'ratio'])
plt.show()

# 変数が1を超えてほしくない場合、ロジスティック関数を使って回帰を行う（直接正規分布を推定しない）
# 1、0のどちらか（ここでは出席、非出席）をとる試行を複数回行った分布は、二項分布を使うと良い。
# ここでは、出席確率をAとScoreの2つの要素から推定する。確率を推定するので、ロジスティック関数を使って推定値qを求める。
# qと授業回数Mを使って、出席した授業回数Yを二項分布に従うものとして推定する。

# データ準備
# スケーリング
Score = attendance_2['Score']/200

Y = attendance_2['Y']
A = attendance_2['A']
M = attendance_2['M']
N = len(Y)


stan_data = {
    'N': N,
    'Score': Score,
    'A': A,
    'Y': Y,
    'M': M
}

# モデルのコンパイル
filename = '../model/model5-4'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)

# 実測値と予測値のプロット
quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
print(np.percentile(mcmc_result['y_pred'], q=quantile, axis=0))
attendance_2_pred = pandas.DataFrame(np.percentile(mcmc_result['y_pred'], q=quantile, axis=0).T, columns=colname)
d = pandas.concat([attendance_2, attendance_2_pred], axis=1)
d0 = d.query('A==0')
d1 = d.query('A==1')

plt.plot([0,0.5], [0,0.5], 'k--', alpha=0.7)
plt.errorbar(d0.Y, d0.p50, yerr=[d0.p50 - d0.p10, d0.p90 - d0.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3)
plt.errorbar(d1.Y, d1.p50, yerr=[d1.p50 - d1.p10, d1.p90 - d1.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='^', mfc='green', capsize=3)
ax = plt.axes()
ax.set_aspect('equal')
plt.show()