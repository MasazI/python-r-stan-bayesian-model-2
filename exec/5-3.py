import numpy as np
import seaborn as sns
import pandas
import mcmc_tools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# ファイルの読み込み
# PersonID: 学生のID
# A: アルバイトが好きかどうか0:好きでない、1:好き
# Score: 学問への興味を数値化したもの
# Weather: 天気(A:晴れ、B:曇り、C:雨)
# Y: 授業に出席したかどうか0:欠席、1:出席
attendance_3 = pandas.read_csv('data-attendance-3.txt')
print(attendance_3.head())
print(attendance_3.describe())

# 解析の目的
# A, Score, Weather を使って、応答変数Yをどのくらい予測できるか

# 2値、カテゴリカル変数が多い場合、散布図ではなく集計でデータを観察すると良い
print(pandas.crosstab(index=attendance_3['Weather'],columns=attendance_3['Y']))

# 練習問題5(3)
print(pandas.crosstab(index=attendance_3['A'],columns=attendance_3['Y']))


# 背景知識を使った変換を実施
# カテゴリカル変数を数値に置き換える際、各カテゴリの応答変数に関する影響度を背景知識として使う方法がある
# ここでは、A、B、Cをそれぞれ0、0.2、1の影響度をもつ変数として変換する。
# 変換用辞書
weather_effec = {
    'A': 0,
    'B': 0.2,
    'C': 1
}

attendance_3_re = attendance_3.replace(weather_effec)
print(attendance_3_re.head())

# 0、1の2値の出現確率を推定したい場合、ベルヌーイ分布を使う
# ここでは、A、Score、WeatherからYを推定する。A、Score、Weatherは出席確率としてロジスティック関数を使って推定し、
# 推定した値qを使って、Yをベルヌーイ分布に従うものとして計算する。
Score = attendance_3_re['Score']/200

Y = attendance_3_re['Y']
A = attendance_3_re['A']
Weather = attendance_3_re['Weather']
N = len(Y)

stan_data = {
    'N': N,
    'Score': Score,
    'A': A,
    'Y': Y,
    'Weather': Weather
}

# コンパイル
filename = '../model/model5-5'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)
mcmc_sample = mcmc_result.extract()

# 取得データの行列を確認
# print(mcmc_sample['q'].shape)
# print(mcmc_sample['q'].T.shape)

q_sample = mcmc_sample['q'].T

# 各qの中央値を取得
q_median = []
for i in range(N):
    q_median.append(np.median(q_sample[i]))

# サンプリングしたqと応答変数の関係を確認
# 書籍とは転地した状態の図を描くほうがPythonでは簡単
# Y0が欠席、Y1が出席なので、qが小さいほどY0、大きいほどY1に振れていれば良い推定といえる。
# 今回はデータが振れていないので良い推定とはいえない。
plot_df = pandas.DataFrame({
    'Y': Y,
    'A': A,
    'q': q_median
})
sns.violinplot(x='Y', y='q', hue='A', data=plot_df)
plt.show()

# ROC曲線を確認
# PythonでROC曲線をえがく場合はskleranが便利
# ROC曲線をプロット
fpr, tpr, thresholds = roc_curve(Y, q_median)

# AUC（ROC曲線の面積）を求めるのもsklearnが便利
auc = auc(fpr, tpr)

# 書籍では閾値を0.6としているがsklearnでは最適な閾値を求める
# 今回の分類器では、AUCが0.62なので、結果は良くない。
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()