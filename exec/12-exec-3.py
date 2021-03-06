# 2次元格子状のマルコフ確率場モデル

# Own Library
import mcmc_tools
import analysis_data as ad

import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


class Markov2Dim(ad.AnalysisData2Dim):
    def create_noisze(self, sigma: float):
        return np.random.normal(self.data, scale=sigma)

    def observe_mesh(self):
        plt.matshow(self.data)
        plt.colorbar(fraction=0.025, pad=0.05)
        plt.xlabel('Plate Column')
        plt.ylabel('Plate Row')
        plt.show()
        plt.close()

    def get_y_max(self):
        return self.data.max().max()

    def create_data(self, noise_sigma: float):
        I = len(self.data)
        J = len(self.data.columns)

        Y = self.data
        T = self.process.max().max()
        TID = self.process

        df_noise = pandas.DataFrame(self.create_noisze(sigma=noise_sigma))
        Y = Y + df_noise

        return {
            'I': I,
            'J': J,
            'Y': Y,
            'T': T,
            'TID': TID
        }, df_noise

    def fit(self, stan_data, save=False):
        mcmc_result = mcmc_tools.sampling(self.model_file, stan_data, n_jobs=6, seed=1234, iter=5200, warmup=200, save=save)
        return mcmc_result.extract()

    def create_figure(self, mcmc_sample, noise_sigma: float):
        I = len(self.data)
        J = len(self.data.columns)
        X, Y = np.mgrid[0:I, 0:J]
        ax = Axes3D(plt.figure(), elev=60)

        # extracting z value
        r = mcmc_sample['r']
        r_median = np.median(r, axis=0)

        ax.plot_wireframe(X, Y, r_median)
        plt.title("r with adding gaussian noise sigma:%f" % (noise_sigma))
        plt.setp(ax, xlabel='Plate Row', ylabel='Plate Column', zlabel='r')
        plt.show()
        plt.close()

    def observe_diff_mean(self, mcmc_sample, noise_sigma: float, noise):
        d = (self.data + noise).values
        TID = self.process.values
        mean_Y = [d[TID == t + 1].mean() - d.mean() for t in range(self.process.max().max())]

        quantile = [2.5, 50, 97.5]
        colname = ['p0025', 'p0500', 'p0975']
        beta = pandas.DataFrame(np.percentile(mcmc_sample['beta'], q=quantile, axis=0).T,
                                  columns=colname)
        beta['x'] = mean_Y
        plt.plot([-5.0, 5.0], [-5.0, 5.0], 'k--', alpha=0.7)
        plt.errorbar(beta.x, beta.p0500, yerr=[beta.p0500 - beta.p0025, beta.p0975 - beta.p0500],
                     fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='green', capsize=3)
        ax = plt.axes()
        ax.set_aspect('equal')
        plt.title("y and beta with adding gaussian noise sigma:%f" % (noise_sigma))
        plt.setp(ax, xlabel='Mean of Y[TID]', ylabel='beat[t]')
        plt.show()
        plt.close()


if __name__ == '__main__':
    m = Markov2Dim('data-2Dmesh.txt', 'data-2Dmesh-design.txt', '../model/model12-exec-3')

    # noiseの大きさごとにサンプリングして結果を確認
    for noise_sigma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        m.create_noisze(noise_sigma)

        # m.describe()
        # m.observe_mesh()
        #
        # # モデリング
        stan_data, df_noise = m.create_data(noise_sigma)
        mcmc_sample = m.fit(stan_data, save=False)

        m.create_figure(mcmc_sample, noise_sigma)
        m.observe_diff_mean(mcmc_sample, noise_sigma, df_noise)
