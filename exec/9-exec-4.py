# model8-2.stanのベクトル化
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


# 年収ファイル3
# X: 年齢、実際からマイナス23（新卒の年齢を23とし、所属年数として扱うため）
# Y: 年収
# KID: 勤務している会社のID(1~30)30社
# GID: 業界のID(1~3)
salary3 = pandas.read_csv('data-salary-3.txt')
print(salary3.head())
print(salary3.describe())

# データの図示
sns.scatterplot(
    x='X',
    y='Y',
    hue='GID',
    data=salary3
)
plt.show()

# モデリング
# 会社ごとに切片と線形結合係数を想定し、それらが全体平均と業界平均から構成されていると考える。
# 業界平均は全体平均を平均とした正規分布から観測されると考える。
# 各会社ごとの切片と線形結合係数は、業界平均を平均とした正規分布から観測されると考える。

Y = salary3['Y']
X = salary3['X']
N = len(Y)
N_group = 30
KID = salary3['KID']
N_industry = 3
GID = salary3['GID']

# 会社IDから業界IDを取得するデータを取得する配列を作成
K2G = salary3.drop_duplicates(subset=['KID', 'GID'])['GID']

stan_data = {
    'Y': Y,
    'X': X,
    'N': N,
    'N_group': N_group,
    'N_industry': N_industry,
    'KID': KID,
    'K2G': K2G,
    'GID': GID
}

# コンパイル
filename = '../model/model-exec9-4'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
mcmc_sample = mcmc_result.extract()