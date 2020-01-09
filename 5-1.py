import numpy as np
import seaborn as sns
import pandas
import mcmc_tools
import matplotlib.pyplot as plt
from scipy.stats import norm

# ファイルの読み込み
attendance_1 = pandas.read_csv('data-attendance-1.txt')
print(attendance_1.head())
print(attendance_1.describe())

# ペアプロットの表示
sns.pairplot(attendance_1, hue="A", diag_kind='hist')
plt.show()

# スケーリング
score = attendance_1['Score']/200
y = attendance_1['Y']
a = attendance_1['A']
sample_num = len(y)

stan_data = {
    'N': sample_num,
    'score': score,
    'a': a,
    'y': y
}

filename = 'model5-3'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4)

# 予測分布
a_range = np.arange(0, 2, 1)
score_range = np.arange(50, 201, 1)
predictions = []
for i in a_range:
    prediction_by_a = []
    for j in score_range:
        mean = mcmc_result['Intercept'].mean() + mcmc_result['b_a'].mean() * i + mcmc_result['b_s'].mean() * j/200
        y = np.random.normal(loc=mean, scale=mcmc_result['sigma'].mean(), size=len(mcmc_result['lp__']))
        prediction_by_a.append(y)
    predictions.append(prediction_by_a)

df_d = pandas.DataFrame(np.array(predictions[0]).T)
df_l = pandas.DataFrame(np.array(predictions[1]).T)
df_d.columns = score_range
df_l.columns = score_range

# visualization 図5.2
observed_dislike = attendance_1.query('A == 0')
observed_like = attendance_1.query('A == 1')
observed = [observed_dislike, observed_like]

label_one = ['dislike', 'like']
qua = [0.025, 0.25, 0.50, 0.75, 0.975]
d_est = pandas.DataFrame()
pallet = ['green', 'blue', 'red']
dataframes = [df_d, df_l]
for j, label in enumerate(label_one):
    for i in score_range:
        for qu in qua:
            d_est[qu] = dataframes[j].quantile(qu)
    x = d_est.index
    y1 = d_est[0.025].values
    y2 = d_est[0.25].values
    y3 = d_est[0.5].values
    y4 = d_est[0.75].values
    y5 = d_est[0.975].values

    plt.fill_between(x,y1,y5,facecolor=pallet[j],alpha=0.1)
    plt.fill_between(x,y2,y4,facecolor=pallet[j],alpha=0.3)
    plt.plot(x,y3,pallet[j],label=label_one[j])
    plt.scatter(observed[j]['Score'], observed[j]['Y'], c=pallet[j])
plt.legend()
plt.show()

print(df_d)

# y_pred = mcmc_result['y_pred']
#
# print(y_pred.shape)
#
# like_index = attendance_1.query('A == 1').index
# dislike_index = attendance_1.query('A == 0').index
#
# y_pred_like = [y_pred[:, i] for i in like_index]
# y_pred_dislike = [y_pred[:, i] for i in dislike_index]
#
# df_l = pandas.DataFrame(y_pred_like)
# df_d = pandas.DataFrame(y_pred_dislike)
#
# print(df_l)

# 実測値と予測値のプロット
quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
print(np.percentile(mcmc_result['y_pred'], q=quantile, axis=0))
attendance_1_pred = pandas.DataFrame(np.percentile(mcmc_result['y_pred'], q=quantile, axis=0).T, columns=colname)
d = pandas.concat([attendance_1, attendance_1_pred], axis=1)
d0 = d.query('A==0')
d1 = d.query('A==1')

plt.plot([0,0.5], [0,0.5], 'k--', alpha=0.7)
plt.errorbar(d0.Y, d0.p50, yerr=[d0.p50 - d0.p10, d0.p90 - d0.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='o', mfc='blue', capsize=3)
plt.errorbar(d1.Y, d1.p50, yerr=[d1.p50 - d1.p10, d1.p90 - d1.p50],
    fmt='o', ecolor='gray', ms=10, alpha=0.8, marker='^', mfc='green', capsize=3)
ax = plt.axes()
ax.set_aspect('equal')
plt.show()

# ノイズの分布確認
print(d.head())


def ip_diff(d):
    return d.loc['Y'] - d.loc['p50']


d['ip'] = d.apply(ip_diff, axis=1)
print(d.head())
# 仮定した正規分布に近い分布になっているか確認
# kdeが密度推定、normが正規分布
# kedとnormが近い形状であれば仮定は問題なかったとする
sns.distplot(d['ip'], bins=13, color='#123456', label='data',
             kde_kws={'label': 'kde', 'color': 'k'},
             fit=norm, fit_kws={'label': 'norm', 'color': 'red'},
             rug=False)
plt.legend()
plt.show()