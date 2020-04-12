# vector同士の内積計算をStanで行う。
import numpy as np
import mcmc_tools

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

ra = np.array([1, 2, 3])

stan_data = {
    'N': 3,
    'a': a,
    'b': b,
    'ra': ra
}

# コンパイル
filename = 'model-exec9-3'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, algorithm='Fixed_param')
mcmc_sample = mcmc_result.extract()

ip = np.inner(a, b)
print(ip)