import os
import pickle
import pystan
import matplotlib.pyplot as plt
import numpy as np
import pandas


def sampling_advi(filename, stan_data, plots=False, save=True):
    if os.path.exists('%s_advi.pkl' % filename):
        sm = pickle.load(open('%s_advi.pkl' % filename, 'rb'))
        # sm = pystan.StanModel(file='%s.stan' % filename)
    else:
        # a model using prior for mu and sigma.
        sm = pystan.StanModel(file='%s.stan' % filename)

    fit_result = sm.vb(data=stan_data, seed=123)

    if plots:
        fit_result.plot()
    if save:
        plt.savefig("%s_advi.png" % filename)
        plt.show()

    # saving compiled model
    if not os.path.exists('%s_advi.pkl' % filename):
        with open('%s_advi.pkl' % filename, 'wb') as f:
            pickle.dump(sm, f)

    return fit_result


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
             pars=None,
             init='random',
             algorithm='NUTS'):
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
            n_jobs=n_jobs,
            algorithm=algorithm,
            init=init
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
            pars=pars,
            algorithm=algorithm,
            init=init
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


def plot_ssm(mcmc_sample, time_vec, title, y_label, state_name, obs_vec=None):
    '''
    plot ssm
    :param mcmc_sample: Dataframe of pandas
    :param time_vec: time vector as x-axis
    :param title: graph title
    :param y_label: y label
    :param state_name: variable name of state
    :param obs_vec: observation's value
    :return:
    '''
    # print(mcmc_sample)
    # print(mcmc_sample[state_name].T.shape)

    # print(np.quantile(mcmc_sample[state_name].T, 0.025))

    mu_df = pandas.DataFrame(mcmc_sample[state_name])

    # print(mu_df.head())

    mu_quantile = mu_df.quantile(q=[0.025, 0.5, 0.975])
    # print(mu_quantile)

    mu_quantile.index = ['lwr', 'fit', 'upr']
    mu_quantile.columns = pandas.Index(time_vec)
    print(mu_quantile)

    y1 = mu_quantile.iloc[0].values
    y2 = mu_quantile.iloc[1].values
    y3 = mu_quantile.iloc[2].values

    plt.fill_between(time_vec, y1, y3, facecolor='blue', alpha=0.1)
    plt.plot(time_vec, y2, 'blue', label=state_name)
    if obs_vec is not None:
        plt.scatter(time_vec, obs_vec, s=5)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('time')
    plt.legend()
    plt.tight_layout()
    plt.show()