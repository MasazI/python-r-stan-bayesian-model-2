# 2次元格子状のマルコフ確率場モデル

# Own Library
import mcmc_tools
import analysis_data as ad

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


class Markov2Dim(ad.AnalysisData2Dim):
    def observe_mesh(self):
        plt.matshow(self.data)
        plt.colorbar(fraction=0.025, pad=0.05)
        plt.xlabel('Plate Column')
        plt.ylabel('Plate Row')
        plt.show()
        plt.close()

    def create_data(self):
        I = len(self.data)
        J = len(self.data.columns)

        Y = self.data
        T = self.process.max().max()
        TID = self.process

        return {
            'I': I,
            'J': J,
            'Y': Y,
            'T': T,
            'TID': TID
        }

    def fit(self, stan_data):
        mcmc_result = mcmc_tools.sampling(self.model_file, stan_data, n_jobs=4, seed=1234, iter=5200, warmup=200)
        return mcmc_result.extract()

    def create_figure(self, mcmc_sample):
        I = len(self.data)
        J = len(self.data.columns)
        X, Y = np.mgrid[0:I, 0:J]
        ax = Axes3D(plt.figure(), elev=60)

        # extracting z value
        r = mcmc_sample['r']
        r_median = np.median(r, axis=0)

        ax.plot_wireframe(X, Y, r_median)
        plt.setp(ax, xlabel='Plate Row', ylabel='Plate Column', zlabel='r')
        plt.show()


if __name__ == '__main__':
    m = Markov2Dim('data-2Dmesh.txt', 'data-2Dmesh-design.txt', 'model12-7')
    m.describe()
    m.observe_mesh()

    # モデリング
    stan_data = m.create_data()
    mcmc_sample = m.fit(stan_data)

    m.create_figure(mcmc_sample)