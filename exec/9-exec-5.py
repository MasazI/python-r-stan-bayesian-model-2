# minとfminの引数の型

# min
# https://mc-stan.org/docs/2_22/functions-reference/reductions.html
# vectorまたはmatrixをとれる

# fmin
# https://mc-stan.org/docs/2_22/functions-reference/step-functions.html
# real型の値を２つとれる

import numpy as np
import mcmc_tools

x = np.array([1, 2, 3])
a = 10
b = 20

stan_data = {
    'N': 3,
    'x': x,
    'a': a,
    'b': b
}

# コンパイル
filename = '../model/model-exec9-5'
mcmc_result = mcmc_tools.sampling(filename, stan_data, n_jobs=4, algorithm='Fixed_param')
mcmc_sample = mcmc_result.extract()