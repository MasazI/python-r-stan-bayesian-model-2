# 時系列予測の予測期間を 8 に増やす
# 時系列予測の問題に季節項を導入する
# 時系列データは、目的変数を観測値の要素の和に分解するのが定石

# Own Library
import mcmc_tools
import analysis_data as ad

import seaborn as sns
import matplotlib.pyplot as plt


class SPM(ad.AnalysisData):
    def observe_ts(self):
        sns.lineplot(x=self.data['X'], y=self.data['Y'])
        plt.show()
        plt.close()

    def create_data(self):
        Y = self.data['Y']
        N = len(Y)
        N_pred = 8
        L = 8

        return {
            'Y': Y,
            'N': N,
            'L': L,
            'N_pred': N_pred
        }

    def fit(self, stan_data):
        mcmc_result = mcmc_tools.sampling(self.model_file, stan_data, n_jobs=4, seed=123)
        return mcmc_result.extract()

    def create_figure(self, mcmc_sample, state: str):
        pred_dates = [i for i in range(len(self.data['Y']) + 8)]
        # pred_dates = np.linspace(0, len(self.data['Y']) + 3, 100)
        mcmc_tools.plot_ssm(mcmc_sample, pred_dates, 'season and trend model (L=8)'
                                                     'model', 'Y', state)


if __name__ == '__main__':
    spm = SPM('data-ss2.txt', '../model/model12-exec-2')
    spm.describe()

    spm.observe_ts()

    stan_data = spm.create_data()

    mcmc_sample = spm.fit(stan_data)

    # 全体の観測および予測分布
    spm.create_figure(mcmc_sample, 'y_mean_pred')

    # 要素ごとに分けて観測および予測分布
    spm.create_figure(mcmc_sample, 'mu_pred')
    spm.create_figure(mcmc_sample, 'season_pred')
