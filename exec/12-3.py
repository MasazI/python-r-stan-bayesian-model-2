# 時系列データにおける変化点検出

# Own Library
import mcmc_tools
import analysis_data as ad

import seaborn as sns
import matplotlib.pyplot as plt


class DetectChangePoint(ad.AnalysisData):
    def observe_ts(self):
        sns.lineplot(x=self.data['X'], y=self.data['Y'])
        plt.show()
        plt.close()

    def create_data(self):
        Y = self.data['Y']
        N = len(Y)
        return {
            'Y': Y,
            'N': N
        }

    def fit(self, stan_data):
        mcmc_result = mcmc_tools.sampling(self.model_file, stan_data, n_jobs=4, seed=123)
        return mcmc_result.extract()

    def create_figure(self, mcmc_sample, state: str):
        # pred_dates = [i for i in range(len(self.data['Y']) + 4)]
        pred_dates = [i for i in range(len(self.data['Y']))]
        mcmc_tools.plot_ssm(mcmc_sample, pred_dates, 'detection about changing point'
                                                     'model', 'Y', state)


if __name__ == '__main__':
    # dcp = DetectChangePoint('data-changepoint.txt', 'model12-3')
    # dcp.describe()
    # dcp.observe_ts()
    # stan_data = dcp.create_data()
    # mcmc = dcp.fit(stan_data)
    # dcp.create_figure(mcmc, 'mu')

    # このままだと、収束がうまくいかずに、
    # 再パラメータ化を勧めるWarningが出てくる。
    # WARNING:pystan:E-BFMI below 0.2 indicates you may need to reparameterize your model
    # そのため、以下のURLを参考に再パラメータ化を行う。
    # https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html
    # 収束しないと出てくるが以外と収束してしまっている、、、

    # 再パラメータ化
    dcp_r = DetectChangePoint('data-changepoint.txt', 'model12-3-re')
    dcp_r.describe()
    dcp_r.observe_ts()
    stan_data = dcp_r.create_data()
    mcmc_r = dcp_r.fit(stan_data)
    dcp_r.create_figure(mcmc_r, 'beta')




