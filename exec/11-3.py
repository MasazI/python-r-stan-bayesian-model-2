# ゼロ過剰ポアソン分布
from typing import List

import pandas
import analysis_data as ad
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import numpy as np

import mcmc_tools


class ZIP(ad.AnalysisData):
    def standardize(self):
        self.data['Age'] = self.data['Age']/10

    def create_stan_data(self):
        Y = self.data['Y']
        N = len(Y)
        X = self.data[['Sex', 'Sake', 'Age']].copy()
        X.insert(0, 'b', 1)
        D = len(X.columns)

        return {
            'Y': Y,
            'N': N,
            'X': X,
            'D': D
        }

    def fit(self, stan_data):
        mcmc_result = mcmc_tools.sampling(self.model_file, stan_data, n_jobs=4, seed=123)
        return mcmc_result.extract()


    def get_data(self, keys: List):
        return self.data[keys]


if __name__ == '__main__':
    # data-ZIP
    # Sex: 0男性、1女性
    # Sake: 0酒をない、1酒をのむ
    # Age: 年齢
    # Y: 来店関数
    z = ZIP('data-ZIP.txt', '../model/model11-3')
    z.describe()

    # z.observe('Sex')
    # 年齢が高いほうが来店回数が多そうだが、全年齢そうに渡って来店していない層がいる。
    # 説明する変数がないだろうか?
    # 来店回数が多いのは男性の割合が多い。

    # z.observe('Sake')
    # 酒を飲まない層が一度も来店しない可能性が高いのではないか？という仮設がもてる。
    # 女性のほうが酒を飲まない割合が高い

    # モデリング
    # 男性、女性とも年齢層に大きな差はないので、年齢とYには相関がありそう。
    # ただし、全年齢層においてYが小さいサンプルが存在するので、
    # 酒を飲む飲まないとYにも相関がありそう。
    # 男性のほうが酒をのむ割合が高いので、Yも男性が多くなっているという仮設が成り立ち、
    # 性別とYの直接の関係は薄いと考える。

    # 最初に重回帰分析を当てはめる
    Y = z.get_data(['Y'])
    X = z.get_data(['Sex', 'Sake', 'Age'])

    lr = LinearRegression()
    lr.fit(X, Y)

    # 順位相関係数
    print(pandas.DataFrame({"Name": X.columns,
                        "Coefficients": lr.coef_[0]}))

    # 解析の目的
    # リピーターになりそうな人を知りたい
    # 説明変数がリポーターになるかにどれほど影響しているかを知りたい
    # 来店客数の分布をみると、0に数字が集中していることと、それ意向は山型の分布
    # つまり、2つの分布から生成されていると考える。
    # 具体的には、
    # とにかく1回足を運んで来店することと、複数回来店することは別の分布だと考える。
    # 1回来店する確率: q（ベルヌーイ分布）
    # リピーターの来店回数: 平均lambdaのポアソン分布

    # q はロジスティック回帰（線形回帰の確率予測版）
    # lambda にはポアソン回帰

    # qとlambdaそれぞれに対して、説明変数と線形結合係数との積を用意する。
    data = z.create_stan_data()

    # Ageの標準化なし
    ms = z.fit(data)

    # Ageの標準化あり（Ageに関連する項目が10倍の重みで出る）
    # z.standardize()
    # data_s = z.create_stan_data()
    # ms_s = z.fit(data_s)

    # qとlamda順位相関
    q = ms['q']
    l = ms['lambda']

    q_m = np.median(q, axis=0)
    l_m = np.median(l, axis=0)

    correlation, pvalue = spearmanr(q_m, l_m)
    print(correlation)
    print(pvalue)

    # 負の相関があり、p値が十分小さい。