# 地図情報を使った空間構造

# Own Library
import mcmc_tools
import analysis_data as ad

import pandas
import seaborn as sns
import japanize_matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from japanmap import pref_names
from japanmap import picture


class Markov2Dim(ad.AnalysisData2Dim):
    def __init__(self, file: str, file2: str, model_file: str):
        self.file = file
        self.data = pandas.read_csv(file)
        self.process = pandas.read_csv(file2)
        self.model_file = model_file

    def create_figure_y(self):
        cmap = plt.get_cmap('binary')
        norm = plt.Normalize(vmin=9.0, vmax=19.0)
        fcol = lambda x: '#' + bytes(cmap(norm(x), bytes=True)[:3]).hex()
        plt.colorbar(plt.cm.ScalarMappable(norm, cmap))
        plt.title('y mean')
        df = self.data['Y'].apply(fcol)

        # indexを振り直す
        df.index = df.index + 1

        plt.imshow(picture(df))
        plt.show()
        plt.close()

    def create_figure(self, mcmc_sample):
        r = mcmc_sample['r'].mean(axis=0)
        # print(r)
        df = pandas.DataFrame(r, columns=['r'])
        # print(df)
        cmap = plt.get_cmap('binary')
        norm = plt.Normalize(vmin=9.0, vmax=19.0)
        fcol = lambda x: '#' + bytes(cmap(norm(x), bytes=True)[:3]).hex()
        plt.colorbar(plt.cm.ScalarMappable(norm, cmap))
        plt.title('r')

        # indexを振り直す
        df = df['r'].apply(fcol)
        df.index = df.index + 1

        plt.imshow(picture(df))
        plt.show()
        plt.close()

    def create_data(self):
        # print(self.data)
        # print(self.process)

        N = len(self.data)
        # print(N)

        Y = self.data['Y']
        To = self.process['To']
        From = self.process['From']

        # がんばったけどこれ不要だった、、、
        # slice = []
        # carsol = 1
        # for i in range(N):
        #     length = len(self.process.query(('From == %d') % (i+1)))
        #     end = carsol+length-1
        #     slice.append((carsol,end))
        #     carsol += length
        # print(slice)
        #
        # for i,(start,end) in enumerate(slice):
        #     print('from pref no: %d' % (i+1))
        #     # print('start:%d, end:%d' % (start-1, end))
        #     to_e = self.process['To'].iloc[start-1:end]
        #     for t in to_e:
        #         y = self.data.query('prefID==%d' % (t))
        #         print('to pref no%d, %f' % (t, y['Y'].values))
        #         print('to pref no%d, %f' % (t, Y[t-1]))

        return {
            'N': N,
            'I': len(To),
            'Y': Y,
            'To': To,
            'From': From
        }

    def fit(self, stan_data):
        mcmc_result = mcmc_tools.sampling(self.model_file,
                                          stan_data,
                                          n_jobs=4,
                                          seed=1234,
                                          init=lambda: dict(r=self.data['Y'], sigma_r=1, sigma_y=0.1)
                                          )
        return mcmc_result.extract()

    def observe_diff_mean(self, mcmc_sample):
        d = self.data['Y'].values
        # print(d)
        quantile = [2.5, 50, 97.5]
        colname = ['p0025', 'p0500', 'p0975']
        df = pandas.DataFrame(np.percentile(mcmc_sample['r'], q=quantile, axis=0).T,
                                  columns=colname)
        df['x'] = d
        plt.plot([9.0, 24.0], [9.0, 24.0], 'k--', alpha=0.7)

        # 県名
        map_jis = pandas.read_csv('data-map-JIS.txt', header=None)
        map_jis.columns = ['ID', 'Name']

        # yとrの差のランキング
        df['diff'] = np.fabs(df['x'] - df['p0500'])
        df['PName'] = map_jis['Name']
        df = df.sort_values('diff', ascending=False)

        for i in range(len(d)):
            if i <= 2:
                plt.text(df.iloc[i].x, df.iloc[i].p0975, df.iloc[i].PName, fontsize=10)
            plt.vlines(x=df.iloc[i].x, ymin=df.iloc[i].p0025, ymax=df.iloc[i].p0975)
            plt.scatter(df.iloc[i].x, df.iloc[i].p0500)

        # plt.errorbar(df.iloc[i].x, df.iloc[i].p0500, yerr=[df.iloc[i].p0500 - df.iloc[i].p0025, df.iloc[i].p0975 - df.iloc[i].p0500],
        #              fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='green', capsize=3)
        ax = plt.axes()
        ax.set_aspect('equal')
        plt.setp(ax, xlabel='Y', ylabel='r')
        plt.show()
        plt.close()


if __name__ == '__main__':
    m = Markov2Dim('data-map-temperature.txt', 'data-map-neighbor.txt',  'model12-8')
    m.describe()

    # Figure of Japan map.
    plt.rcParams['figure.figsize'] = 6, 6
    plt.imshow(picture())
    plt.show()
    plt.close()

    m.create_figure_y()

    # モデリング
    stan_data = m.create_data()
    mcmc_sample = m.fit(stan_data)

    # モデリングした結果の可視化
    m.create_figure(mcmc_sample)

    # 予測と実測のプロット
    m.observe_diff_mean(mcmc_sample)

