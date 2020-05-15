# ここでは、全体の様相を推論に取り込むために階層モデルを使う。
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.markers import MarkerStyle
import mcmc_tools
from scipy.stats import norm
import random

# 年収ファイル2
# X: 年齢、実際からマイナス23（新卒の年齢を23とし、所属年数として扱うため）
# Y: 年収
# KID: 勤務している会社のID(1~4)大手4社

salary2 = pandas.read_csv('data-salary-2.txt')
print(salary2.head())
print(salary2.describe())

# モデリング
# すべての会社で共通の全体平均と会社間のばらつきを分けて考える。
# ばらつきは正規分布から生成さえれると考える。
# この考え方は切片aと各変数の重みbの両方で仮定する。


# 階層モデルではデータをシュミレーションしてモデルの妥当性を確認することがしばしばある。
def sim(debug=False):
    N = 40 # 全体人数
    K = 4 # 会社数
    K_k = np.array([15, 12, 10, 3]) # 各会社の人数
    a0 = 350 # a全体平均
    b0 = 12 # b全体平均
    s_a = 60 # a会社差の標準偏差
    s_b = 4 # b会社差の標準偏差
    s_Y = 25 # 年収の標準偏差

    # 勤続年数をランダムで抽出
    X = random.choices(np.arange(0, 36), k=N)
    # 企業IDを人数に合わせて割り振る
    KID = np.array([], np.int32)
    for i, k in enumerate(K_k):
        KID = np.concatenate([KID, np.array([i + 1 for n in np.arange(k)])], 0)
    # 平均を正規分布からサンプリング
    a = norm.rvs(0, s_a, K) + a0
    b = norm.rvs(0, s_b, K) + b0
    if debug:
        print(a)
        print(b)
        print(X)
        print(KID)
    # データフレームに格納
    df = pandas.DataFrame(
        {'X': X, 'KID': KID, 'a': a[KID-1], 'b': b[KID-1]}
    )
    if debug:
        print(df)

    # 行ごとにYをサンプリング
    def func(row):
        return norm.rvs(row['a'] + row['b']*row['X'], s_Y)

    df['Y'] = df.apply(func, axis=1)
    if debug:
        print(df)

    return df


salary_sim = sim()
print(salary_sim)

# シュミレーションデータのペアプロットの表示
sns.pairplot(salary_sim, hue="KID", diag_kind='hist')
plt.show()
plt.close()


# モデリング
# 線形回帰のパラメータaとbが、全体平均と各会社ごとのノイズで構成されると仮定する。
# モデリングを整理すると、
# 各グループごとにaとbが存在すると考えるのだが、それを全体共通とグループごとに値に分けて考える。
# そのため、aとbをa0、b0という全体共通項と、a1,a2,a3,a4, b1,b2,b3,b4 のグループごとの係数に分ける。
# それらの和を a = a0 + a[N_group] のように記述する。
# また、グループごとの係数が正規分布から生成されたと考える。
# この係数を線形結合した値を平均とした正規分布から生成されるのが、年収Yだと仮定する。
def salary_modeling(salary, filename):
    Y = salary['Y']
    X = salary['X']
    N = len(Y)
    N_group = 4
    KID = salary['KID']

    stan_data = {
        'Y': Y,
        'X': X,
        'N': N,
        'N_group': N_group,
        'KID': KID
    }

    # コンパイル
    mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, seed=123)
    mcmc_sample = mcmc_result.extract()
    return mcmc_sample


def plot(mcmc_sample):
    # 実測値と予測値のプロット
    quantile = [10, 50, 90]
    colname = ['p' + str(x) for x in quantile]
    print(np.percentile(mcmc_sample['Y_pred'], q=quantile, axis=0))
    salary2_pred = pandas.DataFrame(np.percentile(mcmc_sample['Y_pred'], q=quantile, axis=0).T, columns=colname)
    d = pandas.concat([salary2, salary2_pred], axis=1)
    d0 = d.query('KID==1')
    d1 = d.query('KID==2')
    d2 = d.query('KID==3')
    d3 = d.query('KID==4')

    plt.plot([0,800], [0,800], 'k--', alpha=0.7)
    plt.errorbar(d0.Y, d0.p50, yerr=[d0.p50 - d0.p10, d0.p90 - d0.p50],
        fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3)
    plt.errorbar(d1.Y, d1.p50, yerr=[d1.p50 - d1.p10, d1.p90 - d1.p50],
        fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='^', mfc='green', capsize=3)
    plt.errorbar(d2.Y, d2.p50, yerr=[d2.p50 - d2.p10, d2.p90 - d2.p50],
        fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='x', mfc='red', capsize=3)
    plt.errorbar(d3.Y, d3.p50, yerr=[d3.p50 - d3.p10, d3.p90 - d3.p50],
        fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='yellow', capsize=3)

    ax = plt.axes()
    ax.set_aspect('equal')
    plt.legend()
    plt.xlabel('Observed', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.show()
    plt.close()


mcmc_sample_hie = salary_modeling(salary2, '../model/model8-1-hie')
plot(mcmc_sample_hie)

# 描かれたグラフを見ると、単純にグループごとに計算したものよりも
# 推定の信頼区間が広がっていることがわかる。
# 今回の場合はこれが過学習をしづらいことと関連している。

# シュミレーションデータでやってみる。
# mcmc_sample_sim = salary_modeling(salary_sim)
# plot(mcmc_sample_sim)

# シュミレーションデータだと、よくフィットしたモデルができたとは言い難い。
# 4社分のデータしかなかったため、95%信頼区間が大きく設定されている。
# 実際には、年収などの常識的な範囲が設定できるパラメータでは、
# 弱情報事前分布を活用することができる。

a_hie = mcmc_sample_hie['a']

pd_hie = pandas.DataFrame(a_hie)
pd_hie.plot.box()
plt.show()

# 比較
mcmc_sample_t = salary_modeling(salary2, '../model/model10-2-2')
a_group = mcmc_sample_t['a']
pd_group = pandas.DataFrame(a_group)
pd_group.plot.box()
plt.show()
# sigma_aの事前分布にt分布を設定したもの。
# sigma_aのサンプリングにどのような変化が見られるか。
# 95%ベイズ信頼区間がかなり狭くなっている。
# これは新卒の収入の幅は会社内においてたかだか100万円以内である可能性が高い、
# という事前分布を設定したからである。

N_mcmc = mcmc_sample_hie['lp__'].size

# 勤続年数
X_new = np.arange(salary2['X'].max()+1)
N_X = X_new.size

# 会社ごとにデータをサンプリング
K = salary2['KID'].max()
d2 = pandas.DataFrame([])

for k in range(K):
    # 会社ごとのaを取得
    loc_a = mcmc_sample_hie['a'][:, k].reshape((-1, 1))
    # 会社ごとのbを取得して勤続年数と線形結合
    loc_b = np.outer(mcmc_sample_hie['b'][:, k], X_new)
    loc = loc_a + loc_b
    # 分散をlogと同じshapeにブロードキャスト(分散はサンプルごとでKIDに分かれていないので各社で同じ分散を割り当てる)
    scale = np.broadcast_to(mcmc_sample_hie['sigma'].reshape((-1, 1)), loc.shape)
    # データのサンプリング
    print(loc.shape)
    print(scale.shape)
    y_base_mcmc2 = np.random.normal(loc, scale)
    # 予測区間を含めてデータフレーム化
    d2 = d2.append(pandas.DataFrame(np.hstack([X_new.reshape((-1, 1)), np.percentile(y_base_mcmc2, (2.5, 50, 97.5), axis=0).T, np.full_like(X_new, k+1).reshape((-1, 1))]), columns=('X', 'p2.5', 'p50', 'p97.5', 'KID')), ignore_index=True)

print(d2)

# 2x2のaxを準備
_, axes = plt.subplots(2, 2, figsize=figaspect(2/2), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    kid = i + 1
    q = 'KID==@kid'
    ax.fill_between('X', 'p2.5', 'p97.5', data=d2.query(q), color='k', alpha=1/5)
    ax.plot('X', 'p50', data=d2.query(q), color='k', alpha=0.8, label='')
    ax.scatter('X', 'Y', data=salary2.query(q), c='k', marker=MarkerStyle.filled_markers[i], label=kid)
    ax.legend(title='KID')
    plt.setp(ax, title=kid)
plt.tight_layout()
plt.show()