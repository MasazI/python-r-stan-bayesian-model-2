import numpy as np
import seaborn as sns
import pandas
import mcmc_tools
import matplotlib.pyplot as plt
import scipy

# pythonの確率分布関数はscipyが便利
"""
# 練習問題6(1)
"""
## ベルヌーイ分布

# サンプリング
xs = scipy.stats.bernoulli.rvs(p=0.3, size=1000)

x = np.linspace(0,1,2)
p = scipy.stats.bernoulli.pmf(x, 0.2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(xs, bins=np.linspace(-0.5,1.5,3), alpha=0.5, rwidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('frequency')
ax.set_title('histogram')
ax.grid(True)
plt.show()
plt.close()

## カテゴリカル分布
xk = np.arange(7)
pk = (0.1,0.1,0.1,0.1,0.1,0.1,0.4)
custm = scipy.stats.rv_discrete(name='custm', values=(xk, pk))

# サンプリング
xs_c = custm.rvs(size=1000)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(xs_c, bins=np.linspace(0, 6, 7), alpha=0.5, rwidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('frequency')
ax.set_title('histogram')
ax.grid(True)
plt.show()
plt.close()


"""
# 練習問題6(2)
"""
## ベータ分布
# 確率密度関数の描画
x = np.linspace(0,1,100)
for a,b in zip([1,3,5,6,9], [9,6,5,3,1]):
    # 確率密度関数を取得
    beta_pdf = scipy.stats.beta.pdf(x, a, b)
    plt.plot(x,beta_pdf, label='a = {}, b= {}'.format(a,b))
plt.xlim([0,1])
plt.legend(loc='best')
plt.show()
plt.close()


# 発生させた乱数の描画
for a,b in zip([1,3,5,6,9], [9,6,5,3,1]):
    # 乱数を取得
    beta_r = scipy.stats.beta.rvs(a, b, size=100)
    plt.hist(beta_r, density=True, histtype='stepfilled', alpha=0.2, label='a = {}, b= {}'.format(a,b))
plt.xlim([0,1])
plt.legend(loc='best')
plt.show()
plt.close()


## ディリクレ分布
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

xx = np.zeros(shape=[99, 99])
yy = np.zeros(shape=[99, 99])
for a in range(0, 99):
    for b in range(0, 99):
        xx[b][a] = (a + 1) / 100.0 * (100 - (b + 1)) / 100.0
        yy[b][a] = (b + 1) / 100.0

a, b, c = (10, 1, 1)
di = scipy.stats.dirichlet(alpha=[a + 1, b + 1, c + 1])
Z = di.pdf([xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

xx2 = xx + (0.5 - xx.mean(axis=1).reshape(-1, 1))
yy2 = yy * np.sqrt(3) / 2

ax.plot_surface(xx2, yy2, Z)
plt.show()
plt.close()


## ガンマ分布
X = np.arange(0,7,0.1)
for a in [1, 2, 3]:
    for b in [0.5, 1, 2]:
        gd = scipy.stats.gamma.pdf(X, a, scale=1.0 / b)
        plt.plot(X, gd, label=f'Gamma({a}, {b})', color=plt.get_cmap('tab10')(a), linewidth=b)
plt.legend()
plt.ylim(0)
plt.xlim(0)
plt.show()
plt.close()


## 2変量正規分布
x,y = np.meshgrid(np.linspace(-10,10,100),np.linspace(-10,10,100))
pos = np.dstack((x,y))
# 平均
mean = np.array([2.5, 3.3])
# 共分散行列
cov  = np.array([[1.0,0.0],[0.0,1.0]])
# 多変量正規分布を取得
z = scipy.stats.multivariate_normal(mean,cov).pdf(pos)

fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
ax.contourf(x,y,z)
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('pdf')
plt.show()
plt.close()


## コーシー分布
mean, var, skew, kurt = scipy.stats.cauchy.stats(moments='mvsk')
# 平均や分散がすべて未定義であることを確認
print(mean, var, skew, kurt)

# パーセント点関数を観察してみる、0.01区間
print(scipy.stats.cauchy.ppf(0.01))

# パーセント点関数を観察してみる、0.99区間
print(scipy.stats.cauchy.ppf(0.99))

# パーセント転換数に合わせてx軸を作成
x = np.linspace(scipy.stats.cauchy.ppf(0.01),scipy.stats.cauchy.ppf(0.99), 100)
plt.plot(x, scipy.stats.cauchy.pdf(x),'r-', lw=5, alpha=0.6, label='cauchy pdf')
plt.show()
plt.close()


"""
# 練習問題6(3)
"""
# 確率変数
# y_1 mean 50, sv 20
# y_2 mean 20, sv 15
y_1 = scipy.stats.norm.rvs(loc=50, scale=20, size=2000)
y_2 = scipy.stats.norm.rvs(loc=20, scale=15, size=2000)

y = y_1 - y_2

# この形状は混合正規分布ではなく、正規分布になる。
# 正規分布の確率変数は加法性をもち、演算後の確率変数も正規分布に従うことが知られる。
plt.hist(y, density=True, histtype='stepfilled', alpha=0.2)
plt.show()
plt.close()

"""
# 練習問題6(4)
"""
# ここではχ2乗分布を描いてみる。
# 期待値からのズレを表現できるため、検定によく使われる。
X = np.arange(0, 10, 0.01)
for k in range(1, 10):
    plt.plot(X, scipy.stats.chi2.pdf(X, k), label=f'Chi({k})')
    plt.axvline(x=k, color=plt.get_cmap('tab10')(k - 1), linewidth=0.5)
plt.ylim(0, 1)
plt.xlim(0, 10)
plt.legend()
plt.show()