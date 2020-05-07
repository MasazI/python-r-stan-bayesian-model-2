# 状態空間モデルとマルコフ場モデルは等価である
# 直感的な説明は非常にわかりやすい
# Own Library
import mcmc_tools
import analysis_data as ad

import seaborn as sns
import matplotlib.pyplot as plt


class Markov(ad.AnalysisData):
    def observe_ts(self):
        sns.lineplot(x=[i for i in range(len(self.data['Y']))], y=self.data['Y'])
        plt.show()
        plt.close()

    def create_data(self):
        Y = self.data['Y']
        I = len(Y)
        return {
            'Y': Y,
            'I': I
        }

    def fit(self, stan_data):
        mcmc_result = mcmc_tools.sampling(self.model_file, stan_data, n_jobs=4, seed=123)
        return mcmc_result.extract()

    def create_figure(self, mcmc_sample, state: str):
        # pred_dates = [i for i in range(len(self.data['Y']) + 4)]
        pred_dates = [i for i in range(len(self.data['Y']))]
        mcmc_tools.plot_ssm(mcmc_sample, pred_dates, 'markov model', 'Y', state)


if __name__ == '__main__':
    m = Markov('data-kubo11a.txt', 'model12-6')
    m.describe()
    m.observe_ts()

    stan_data = m.create_data()
    mcmc_sample = m.fit(stan_data)

    m.create_figure(mcmc_sample, 'Y_mean')