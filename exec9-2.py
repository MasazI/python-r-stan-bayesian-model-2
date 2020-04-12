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

# ここでの解析の目的は、AとScoreがMにどの程度寄与しているかを判断したいとする。
# 離散した数値の推定はポアソン分布による近似が第一選択。
# 説明変数は、AとScore

# データ準備
# スケーリング
Score = attendance_2['Score']/200

A = attendance_2['A']
M = attendance_2['M']
N = len(M)

stan_data = {
    'N': N,
    'Score': Score,
    'A': A,
    'M': M
}

# コンパイル
filename = 'model-exec9-2'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
mcmc_sample = mcmc_result.extract()

# ポアソン分布を利用すると、
# 推定の結果から、各説明変数を変化させた際に応答変数がどのように変化するか、
# 例えば、Scoreが50から150になった際に、履修登録する授業回数が何倍になるか、
# という計算が簡単な指数の計算で求まる。

# 演習問題 5(5)
# 実測値と予測値のプロット
# 説明変数はAとScoreだが、ここでは応答変数とそれに対する予測値のみを扱う。もち完全に予測できれいれば、
# すべて同じ数値となるが、予測値はサンプリングしているので信頼区間として表示することになる。
quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
print(np.percentile(mcmc_result['m_pred'], q=quantile, axis=0))
attendance_2_pred = pandas.DataFrame(np.percentile(mcmc_result['m_pred'], q=quantile, axis=0).T, columns=colname)
# 実装値と予測値のDataFrameを作成
d = pandas.concat([attendance_2, attendance_2_pred], axis=1)
d0 = d.query('A==0')
d1 = d.query('A==1')

plt.plot([0,0.5], [0,0.5], 'k--', alpha=0.7)
plt.errorbar(d0.M, d0.p50, yerr=[d0.p50 - d0.p10, d0.p90 - d0.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3, label="A=0")
plt.errorbar(d1.M, d1.p50, yerr=[d1.p50 - d1.p10, d1.p90 - d1.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='^', mfc='green', capsize=3, label="A=1")
ax = plt.axes()
ax.set_aspect('equal')
plt.title('model-exec9-2')
plt.legend()
plt.xlabel('Observed', fontsize=20)
plt.ylabel('Predicted', fontsize=20)
plt.show()

# 今回は、ほとんどの信頼区間がy=xを含んでいないことから、実際の値と比べて予測値が離れてしまっていることがわかる。
