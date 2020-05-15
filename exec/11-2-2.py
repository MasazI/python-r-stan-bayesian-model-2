# K個の正規混合分布
import importlib
from typing import Dict

# Onw library
import mcmc_tools

lib = importlib.import_module("11-2")


class MembersK(lib.Members):
    def create_stan_data(self) -> Dict[str, str]:
        Y = self.mix1['Y']
        N = len(Y)

        stan_data = {
            'Y': Y,
            'N': N,
            'K': 5
        }
        return stan_data

    def fit(self, stan_data: Dict[str, str], init: Dict):
        mcmc_result = mcmc_tools.sampling(self.model_file, stan_data, n_jobs=4, seed=123)
        mcmc_sample = mcmc_result.extract()
        return mcmc_sample


if __name__ == '__main__':
    m = MembersK('data-mix2.txt', '../model/model11-2-2')
    m.describe()

    # いくつかの正規分布が混合していると考えられる
    m.hist()

    d = m.create_stan_data()

    # K混合正規分布は
    m.fit(d, {})

    # 同じことを、functionsブロックというstanの関数を独自定義できる機能を使って書き換える
    mb = MembersK('data-mix2.txt', '../model/model11-2-2b')
    db = mb.create_stan_data()
    m.fit(db, {})
    # 同様の結果であることを確認する。
