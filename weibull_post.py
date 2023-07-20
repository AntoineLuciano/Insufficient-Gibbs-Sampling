import numpy as np
import scipy


def weibull_pdf(x, loc, scale, k):
    return (
        k
        / scale
        * ((x - loc) / scale) ** (k - 1)
        * np.exp(-(x - loc) * k / scale)
        * (x > loc)
    )


def weibull_log_pdf(x, loc, scale, k):
    return (
        np.log(k)
        - np.log(scale)
        + (k - 1) * np.log((x - loc) / scale)
        - (x - loc) * k / scale
    )


def weibull_llike_scale(x, loc, scale, shape):
    N = len(x)
    return -N * shape * np.log(scale) - np.sum(((x - loc) / scale) ** shape)


def weibull_llike_shape(x, loc, scale, shape):
    N = len(x)
    return (
        -N * shape * np.log(scale)
        - np.sum(((x - loc) / scale) ** shape)
        + N * np.log(shape)
        + (shape - 1) * np.sum(np.log(x - loc))
    )


def weibull_llike_loc(x, loc, scale, shape):
    return -np.sum(((x - loc) / scale) ** shape) + (shape - 1) * np.sum(np.log(x - loc))


def weibull_log_like(x, loc, scale, k):
    N = len(x)
    # print("ici",loc,scale,k,N,np.min(x),np.min(x-loc),np.log(x-loc))
    # return N*(np.log(k)-k*np.log(scale))+(k-1)*np.sum(np.log(x-loc))-np.sum(((x-loc)/scale)**k)

    return np.sum(np.log(scipy.stats.weibull_min.pdf(x, c=k, scale=scale, loc=loc)))


def post_weibull_loc(loc, scale, k, x, par_prior, std_prop):
    def log_prior_loc(loc, par_prior):
        mu_0, sigma_0 = par_prior[0], par_prior[1]
        return -((loc - mu_0) ** 2) / sigma_0**2 / 2

    loc_star = np.random.normal(loc=loc, scale=std_prop)

    if loc_star >= np.min(x):
        return loc

    current_llikelihood = weibull_log_like(x, loc, scale, k)
    candidate_llikelihood = weibull_log_like(x, loc_star, scale, k)

    current_lprior = log_prior_loc(loc, par_prior)
    candidate_lprior = log_prior_loc(loc_star, par_prior)

    ratio_acceptation = min(
        np.exp(
            candidate_llikelihood
            - current_llikelihood
            + candidate_lprior
            - current_lprior
        ),
        1,
    )

    if np.random.uniform() < ratio_acceptation:
        return loc_star
    return loc


def log_gamma_pdf(x, a, b):
    return (a - 1) * np.log(x) - x * b


def post_weibull_loc2(loc, scale, k, x, par, std_prop):
    loc_star = np.random.normal(loc, std_prop)
    if loc_star <= 0:
        return loc

    current_llikelihood = weibull_log_like(x, loc, scale, k)
    candidate_llikelihood = weibull_log_like(x, loc_star, scale, k)

    current_lprior = log_gamma_pdf(loc, par[0], par[1])
    candidate_lprior = log_gamma_pdf(loc_star, par[0], par[1])

    ratio_acceptation = min(
        np.exp(
            candidate_llikelihood
            - current_llikelihood
            + candidate_lprior
            - current_lprior
        ),
        1,
    )

    if np.random.uniform() < ratio_acceptation:
        return loc_star
    return loc


def post_weibull_scale(loc, scale, k, x, par, std_prop):
    scale_star = np.random.normal(scale, std_prop)
    if scale_star <= 0:
        return scale

    current_llikelihood = weibull_log_like(x, loc, scale, k)
    candidate_llikelihood = weibull_log_like(x, loc, scale_star, k)

    # current_llikelihood = weibull_llike_scale(x,loc,scale,k)
    # candidate_llikelihood = weibull_llike_scale(x,loc,scale_star,k)
    current_lprior = log_gamma_pdf(scale, par[0], par[1])
    candidate_lprior = log_gamma_pdf(scale_star, par[0], par[1])

    # current_lprior =-np.log(scale)
    # candidate_lprior = -np.log(scale_star)

    ratio_acceptation = min(
        np.exp(
            candidate_llikelihood
            - current_llikelihood
            + candidate_lprior
            - current_lprior
        ),
        1,
    )
    # print("SCALE : \ncurrent = {}, llike = {} lprior = {}\ncandidate = {},llike = {} lprior = {}\nDiff = {} Ratio = {}".format(scale,current_llikelihood,current_lprior,scale_star,candidate_llikelihood,candidate_lprior,candidate_llikelihood - current_llikelihood+ candidate_lprior- current_lprior,ratio_acceptation))
    # print("Ratio scale =",ratio_acceptation,end=" ")
    if np.random.uniform() < ratio_acceptation:
        return scale_star
    return scale


def post_weibull_k(loc, scale, k, x, par, std_prop):
    k_star = np.random.normal(k, std_prop)
    if k_star <= 0:
        return k

    current_llikelihood = weibull_log_like(x, loc, scale, k)
    candidate_llikelihood = weibull_log_like(x, loc, scale, k_star)

    # current_llikelihood = weibull_llike_shape(x,loc,scale,k)
    # candidate_llikelihood = weibull_llike_shape(x,loc,scale,k_star)

    # current_lprior =-np.log(k)
    # candidate_lprior = -np.log(k_star)

    current_lprior = log_gamma_pdf(k, par[0], par[1])
    candidate_lprior = log_gamma_pdf(k_star, par[0], par[1])

    ratio_acceptation = min(
        np.exp(
            candidate_llikelihood
            - current_llikelihood
            + candidate_lprior
            - current_lprior
        ),
        1,
    )
    # print("Ratio k =",ratio_acceptation)
    # print("K : \ncurrent = {}, llike = {} lprior = {}\ncandidate = {},llike = {} lprior = {}\nDiff = {} Ratio = {}".format(k,current_llikelihood,current_lprior,k_star,candidate_llikelihood,candidate_lprior,candidate_llikelihood - current_llikelihood+ candidate_lprior- current_lprior,ratio_acceptation))
    if np.random.uniform() < ratio_acceptation:
        return k_star
    return k
