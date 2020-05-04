# 混合正規分布
# 社員の能力測定の結果を利用
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import mcmc_tools
from scipy.stats import norm
from scipy.stats import binom
import random
from typing import Dict


class Members():
    def __init__(self, file: str, model_file: str):
        self.file = file
        self.mix1 = pandas.read_csv(file)
        self.model_file = model_file

    def describe(self):
        print(self.mix1.head())
        print(self.mix1.describe())

    def hist(self):
        ax = self.mix1.plot.hist(bins=30, alpha=0.5, legend=True)
        self.mix1.plot.kde(legend=True, alpha=0.5, secondary_y=True, ax=ax)
        plt.show()

    def create_stan_data(self) -> Dict[str, str]:
        Y = self.mix1['Y']
        N = len(Y)

        stan_data = {
            'Y': Y,
            'N': N
        }
        return stan_data

    def fit(self, stan_data: Dict[str, str], init: Dict) -> pandas.DataFrame:
        mcmc_result = mcmc_tools.sampling(self.model_file, stan_data, n_jobs=4, seed=123,init=init)
        mcmc_sample = mcmc_result.extract()
        return mcmc_sample


if __name__ == '__main__':
    # 社員の能力測定の結果
    m = Members('data-mix1.txt', 'model11-2')
    m.describe()

    # 可視化、混合正規分布であることを確認
    m.hist()

    data = m.create_stan_data()
    print(data)

    # 混合正規分布など複雑なモデルは初期値を与えないと収束しないことが多い。
    # このモデルもそう。
    init = lambda: dict(
        a=0.5,
        mu=(0, 5),
        sigma=(1, 1)
    )
    m.fit(data, init)



