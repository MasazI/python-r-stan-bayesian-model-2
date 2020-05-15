# Latent Dirichlet Allocation (LDA)
# General Library
import math

# Library
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pystan

# Own Library
import mcmc_tools
import analysis_data as ad


class LDA(ad.AnalysisData):
    def observe_joint(self):
        sns.jointplot(self.data['PersonID'], self.data['ItemID'])
        plt.show()
        plt.close()

    def observe_matshow(self):
        im = plt.matshow(pandas.crosstab(self.data['PersonID'], self.data['ItemID']), cmap='binary', aspect='equal')
        plt.colorbar(im, fraction=0.02, pad=0.03)
        plt.setp(plt.gca(), xlabel='ItemID', ylabel='PersonID')
        plt.show()
        plt.close()

    def hist_group(self, key: str):
        self.data.groupby([key]).count().hist(bins=30)
        plt.xlabel("count by %s" % key)
        plt.show()
        plt.close()

    def create_data2(self):
        # 同じ人とItemIDで集計してモデリングする
        df_grouped_s = self.data.groupby(['PersonID', 'ItemID']).size()
        df_grouped = pandas.DataFrame(df_grouped_s)

        df_grouped = df_grouped.reset_index()

        ItemID = df_grouped['ItemID']
        PersonID = df_grouped['PersonID']
        E = len(ItemID)
        K = 6
        I = df_grouped['ItemID'].max()
        N = df_grouped['PersonID'].max()
        C = df_grouped[0]

        stan_data = {
            'E': E,
            'N': N,
            'K': K,
            'I': I,
            'C': C,
            'PersonID': PersonID,
            'ItemID': ItemID,
            'Alpha': np.repeat(0.5, I)
        }
        return stan_data

    def create_data(self):
        ItemID = self.data['ItemID']
        PersonID = self.data['PersonID']
        E = len(ItemID)
        K = 6
        I = self.data['ItemID'].max()
        N = self.data['PersonID'].max()

        stan_data =  {
            'E': E,
            'N': N,
            'K': K,
            'I': I,
            'PersonID': PersonID,
            'ItemID': ItemID,
            'Alpha': np.repeat(0.5, I)
        }
        return stan_data

    def fit(self, stan_data):
        mcmc_result = mcmc_tools.sampling(self.model_file, stan_data, n_jobs=4, seed=123)
        return mcmc_result.extract()

    def fit_advi(self, stan_data):
        advi_result = mcmc_tools.sampling_advi(self.model_file, stan_data)
        return advi_result

    @staticmethod
    def observe_matshow_by_df(df: pandas.DataFrame):
        im = plt.matshow(pandas.crosstab(df['PersonID'], df['ItemID']), cmap='binary', aspect='equal')
        plt.colorbar(im, fraction=0.02, pad=0.03)
        plt.setp(plt.gca(), xlabel='ItemID', ylabel='PersonID')
        plt.show()
        plt.close()

    @staticmethod
    def simulate(n: int, k: int, e: int):
        # ディリクレ分布から生成Kクラスのカテゴリカル分布を生成
        alpha0 = np.array([0.8 for i in range(k)])
        theta = stats.dirichlet.rvs(alpha0, size=n)

        alpha1 = np.array([0.2 for i in range(e)])
        phi = stats.dirichlet.rvs(alpha1, size=n)

        num_item_by_n = list(map(lambda x: math.floor(math.exp(x)), stats.norm.rvs(loc=2.0, scale=0.5, size=n)))

        df = pandas.DataFrame()
        for i in range(n):
            z = np.random.choice(k, size=num_item_by_n[i], replace=True, p=theta[i])
            item = list(map(lambda x: int(np.random.choice(e, size=1, p=phi[x])), z))
            df_raw = pandas.DataFrame(
                {'PersonID': i, 'ItemID': item}
            )
            df = pandas.concat([df, df_raw])
        return df

    @staticmethod
    def visualize(ms, N: int, K: int, I: int):
        probs = (10, 25, 50, 75, 90)
        # 120面サイコロ6type分のインデックス
        idx = np.array([[k + 1, i + 1] for k, i in np.ndindex(K, I)])

        # 720（120面サイコロを6type） x 5 (percentile 10, 25, 50, 75, 90)
        d_qua = np.array([np.percentile(ms['phi.{}.{}'.format(k, i)], probs) for k, i in idx])

        # 上記データにtagとitem、各パーセンタイル値のカラムをつけてDataFrameに格納
        d_qua = pandas.DataFrame(np.hstack((idx, d_qua)), columns=['tag', 'item'] + ['p{}'.format(p) for p in probs])

        print(d_qua)

        # タグごとに商品ごとの出現確率の中央値(p50)を表示する
        for i in range(1,7):
            print('Type: %d' % i)
            d_qua_by_tag = d_qua.query('tag == %f' % float(i))

            d_p50 = d_qua_by_tag[['item', 'p50']]
            print(d_p50)
            ax = sns.barplot(x="item", y="p50", data=d_p50,
                        label="Type %d" % i, color="b", orient='v')
            xlabel = np.arange(0, 121, 20)
            plt.setp(ax, xticks=xlabel, xticklabels=xlabel, xlim=(1, 120), ylim=(0.0, 0.1), xlabel='Item', ylabel='phi[k, y]')
            plt.show()
            plt.close()

        # 顧客ごとに求めたサイコロを表示
        # 顧客数 x 面の数のindex
        idx = np.array([[n + 1, k + 1] for n, k in np.ndindex(N, K)])

        d_qua = np.array([np.percentile(ms['theta.{}.{}'.format(n, k)], probs) for n, k in idx])
        # 上記データにpersonとtag、各パーセンタイル値のカラムをつけてDataFrameに格納
        d_qua = pandas.DataFrame(np.hstack((idx, d_qua)), columns=['person', 'tag'] + ['p{}'.format(p) for p in probs])
        for i, person in enumerate([1, 50]):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            d_qua_by_person = d_qua.query('person == %f' % float(person))
            print(d_qua_by_person)
            ax.barh('tag', 'p50', data=d_qua_by_person, xerr=(d_qua_by_person['p25'], d_qua_by_person['p75']), color='w', edgecolor='k')
            plt.setp(ax, title=person, ylabel='tag')
            plt.show()
        plt.close()


if __name__ == '__main__':
    lda = LDA('data-lda.txt', '../model/model11-exec-6')
    lda.describe()
    data = lda.create_data2()

    advi_result = lda.fit_advi(data)
    ms = pandas.read_csv(advi_result['args']['sample_file'].decode('utf-8'), comment='#')
    print(ms)

    #
    # # モデリングの目的は、購入履歴から顧客の特徴を抽出したい、商品をグルーピングしたい
    # lda.observe_matshow()
    #
    # lda.hist_group('PersonID')
    # lda.hist_group('ItemID')
    #
    # df = lda.simulate(50, 6, 120)
    # lda.observe_matshow_by_df(df)
    #
    # # さて、モデリング。
    # data = lda.create_data()
    # # lda.fit(data)
    #
    # # 変分ベイズの1種、ADVI(Automatic Differentiation Variational Inference)
    # advi_result = lda.fit_advi(data)
    # ms = pandas.read_csv(advi_result['args']['sample_file'].decode('utf-8'), comment='#')
    # print(ms)
    #
    # 結果のビジュアライズ
    # K個のトピック（タグ）ごとの各商品の出現確率の中央値を表示する
    lda.visualize(ms, 50, 6, 120)
