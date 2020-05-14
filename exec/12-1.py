# Own Library
import mcmc_tools
import analysis_data as ad

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class SPM(ad.AnalysisData):
    def observe_ts(self):
        sns.lineplot(x=self.data['X'], y=self.data['Y'])
        plt.show()
        plt.close()

    def create_data(self):
        X = self.data['X']
        Y = self.data['Y']
        N = len(Y)
        N_pred = 3

        return {
            'X': X,
            'Y': Y,
            'N': N,
            'N_pred': N_pred
        }

    def fit(self, stan_data):
        mcmc_result = mcmc_tools.sampling(self.model_file, stan_data, n_jobs=4, seed=123)
        return mcmc_result.extract()

    def create_figure(self, mcmc_sample):
        pred_dates = [i for i in range(len(self.data['Y']) + 3)]
        # pred_dates = np.linspace(0, len(self.data['Y']) + 3, 100)
        mcmc_tools.plot_ssm(mcmc_sample, pred_dates, 'local level '
                                                     'model', 'Y', 'mu_pred')


if __name__ == '__main__':
    spm = SPM('data-ss1.txt', 'model12-1')
    spm.describe()

    spm.observe_ts()

    stan_data = spm.create_data()
    mcmc_sample = spm.fit(stan_data)
    spm.create_figure(mcmc_sample)



