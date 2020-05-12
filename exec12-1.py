# 時系列モデルのシュミレーション
import numpy as np
import pandas
import matplotlib.pyplot as plt


def simurate(T:int, sigma_mu, sigma_y):
    value = np.zeros(T)
    mu = np.zeros(T)
    mu[0] = value[0]
    for i in range(1, T):
        mu[i] = np.random.normal(mu[i-1], sigma_mu)
        value[i] = np.random.normal(mu[i], sigma_y)
    df = pandas.DataFrame(value, columns=['y'])
    df.plot()
    plt.show()


if __name__ == '__main__':
    T = 1000
    # 前後の時刻との変化が多いが、全体のトレンドは似ている
    simurate(T, 0.1, 2.0)
    # 前後の時刻との変化は小さいが、全体のご連弩は大きく変化する
    simurate(T, 2.0, 0.1)
