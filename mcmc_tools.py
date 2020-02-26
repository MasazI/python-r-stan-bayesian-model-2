import os
import pickle
import pystan
import matplotlib.pyplot as plt


def sampling(filename,
             stan_data,
             adapt_delta=0.8,
             max_treedepth=10,
             chains=4,
             iter=2000,
             warmup=1000,
             n_jobs=4,
             save=True,
             seed=1,
             plots=False,
             pars=None):
    """
    Sampling using mcmc with PyStan.
    :param filename:
    :param stan_data:
    :param adapt_delta:
    :param max_treedepth:
    :return:
    """
    if os.path.exists('%s.pkl' % filename):
        sm = pickle.load(open('%s.pkl' % filename, 'rb'))
        # sm = pystan.StanModel(file='%s.stan' % filename)
    else:
        # a model using prior for mu and sigma.
        sm = pystan.StanModel(file='%s.stan' % filename)

    control = {
        'adapt_delta': adapt_delta,
        'max_treedepth': max_treedepth
    }

    if pars is None:
        mcmc_result = sm.sampling(
            data=stan_data,
            seed=seed,
            chains=chains,
            iter=iter,
            warmup=warmup,
            control=control,
            thin=6,
            n_jobs=n_jobs
        )
    else:
        mcmc_result = sm.sampling(
            data=stan_data,
            seed=seed,
            chains=chains,
            iter=iter,
            warmup=warmup,
            control=control,
            thin=6,
            n_jobs=n_jobs,
            pars=pars
        )

    print(mcmc_result)
    if plots:
        mcmc_result.plot()
    if save:
        plt.savefig("%s.png" % filename)
        plt.show()

    # saving compiled model
    if not os.path.exists('%s.pkl' % filename):
        with open('%s.pkl' % filename, 'wb') as f:
            pickle.dump(sm, f)

    return mcmc_result
