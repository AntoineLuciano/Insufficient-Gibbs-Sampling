import numpy as np
import scipy
from scipy.stats import norm, cauchy, weibull_min
from tqdm import tqdm

from truncated import *

# from posterior_sample import *

import numpy as np
from scipy.stats import cauchy, norm, gamma, weibull_min, gamma


def posterior(
    X,
    theta,
    distribution,
    prior_loc,
    prior_scale,
    prior_shape,
    par_prior_loc,
    par_prior_scale,
    par_prior_shape,
    std_prop_loc,
    std_prop_scale,
    std_prop_shape,
):
    def post_NIG(X, par_prior_loc, par_prior_scale):
        mu_0, nu = par_prior_loc
        alpha, beta = par_prior_scale
        n = len(X)
        tau = np.random.gamma(
            shape=alpha + n / 2,
            scale=1
            / (
                beta
                + np.sum((X - np.mean(X)) ** 2) / 2
                + n * nu * (np.mean(X) - mu_0) ** 2 / (2 * (nu + n))
            ),
            size=1,
        )[0]
        mu = np.random.normal(
            loc=(nu * mu_0 + np.sum(X)) / (nu + n),
            scale=1 / np.sqrt((nu + n) * tau),
            size=1,
        )[0]
        return [mu, 1 / np.sqrt(tau), 0]

    def llike_cauchy(x, loc, scale, shape):
        return np.sum(np.log(cauchy.pdf(x, loc=loc, scale=scale)))

    def llike_normal(x, loc, scale, shape):
        return np.sum(np.log(norm.pdf(x, loc=loc, scale=scale)))

    def llike_weibull(x, loc, scale, shape):
        return np.sum(np.log(weibull_min.pdf(x, c=shape, scale=scale, loc=loc)))

    if distribution == "normal" and prior_loc == "NIG":
        return post_NIG(X, par_prior_loc, par_prior_scale)
    loc, scale, shape = theta
    if distribution == "normal":
        llike = llike_normal
    elif distribution == "cauchy":
        llike = llike_cauchy
    elif distribution == "weibull":
        llike = llike_weibull
    elif distribution == "translated_weibull":
        llike = llike_weibull
    else:
        raise Exception("ERROR : Distribution {} not valid !".format(distribution))
    # print("Minimum X = {} loc = {}".format(np.min(X), loc))

    # METROPOLIS HASTINGS STEP FOR LOCATION PARAMETER
    if distribution != "weibull":
        if prior_loc == "cauchy":
            lprior_loc = lambda x: np.log(
                cauchy(loc=par_prior_loc[0], scale=par_prior_loc[1]).pdf(x)
            )
        elif prior_loc == "normal":
            lprior_loc = lambda x: np.log(
                norm(loc=par_prior_loc[0], scale=par_prior_loc[1]).pdf(x)
            )
        elif prior_loc == "gamma":
            if par_prior_loc[0] <= 0 or par_prior_loc[1]:
                raise Exception(
                    "ERROR : prior location parameter invalid for gamma prior !"
                )
            lprior_loc = lambda x: np.log(
                gamma(a=par_prior_loc[0], scale=par_prior_loc[1]).pdf(x)
            )
        else:
            raise Exception(
                "ERROR : Prior for location {} not valid !".format(prior_loc)
            )

        loc_star = np.random.normal(loc, std_prop_loc)
        # print("loc = {}, loc_star = {}, min(X) = {}".format(loc, loc_star,np.min(X)))
        if not (
            (loc_star >= np.min(X) and distribution == "translated_weibull")
            or (loc_star <= 0 and prior_loc != "gamma")
            or (loc_star <= 0 and distribution == "weibull")
        ):
            current_llike, candidate_llike = llike(X, loc, scale, shape), llike(
                X, loc_star, scale, shape
            )

            current_lprior, candidate_lprior = lprior_loc(loc), lprior_loc(loc_star)

            ratio_acceptation = min(
                np.exp(
                    candidate_llike - current_llike + candidate_lprior - current_lprior
                ),
                1,
            )

            if np.random.uniform() < ratio_acceptation:
                loc = loc_star

    # METROPOLIS HASTINGS STEP FOR SCALE PARAMETER
    if prior_scale == "gamma":
        lprior_scale = lambda x: np.log(
            gamma(a=par_prior_scale[0], scale=par_prior_scale[1]).pdf(x)
        )
    elif prior_scale == "jeffrey":
        lprior_scale = lambda x: -np.log(x)
    else:
        raise Exception("ERROR : Prior for scale {} not valid !".format(prior_scale))
    scale_star = np.random.normal(scale, std_prop_scale)

    if scale_star > 0:
        current_llike, candidate_llike = llike(X, loc, scale, shape), llike(
            X, loc, scale_star, shape
        )
        current_lprior, candidate_lprior = lprior_scale(scale), lprior_scale(scale_star)

        ratio_acceptation = min(
            np.exp(candidate_llike - current_llike + candidate_lprior - current_lprior),
            1,
        )

        if np.random.uniform() < ratio_acceptation:
            scale = scale_star

    # METROPOLIS HASTINGS STEP FOR SHAPE PARAMETER

    if distribution in ["translated_weibull", "weibull"]:
        if prior_shape == "gamma":
            lprior_shape = lambda x: np.log(
                gamma(a=par_prior_shape[0], scale=par_prior_shape[1]).pdf(x)
            )
        elif prior_shape == "jeffrey":
            lprior_shape = lambda x: -np.log(x)
        else:
            raise Exception(
                "ERROR : Prior for shape '{}' not valid !".format(prior_shape)
            )

        shape_star = np.random.normal(shape, std_prop_shape)

        if shape_star > 0:
            current_llike, candidate_llike = llike(X, loc, scale, shape), llike(
                X, loc, scale, shape_star
            )

            current_lprior, candidate_lprior = lprior_shape(shape), lprior_shape(
                shape_star
            )

            ratio_acceptation = min(
                np.exp(
                    candidate_llike - current_llike + candidate_lprior - current_lprior
                ),
                1,
            )

            if np.random.uniform() < ratio_acceptation:
                shape = shape_star
        # print("shape = {}, shape_star = {}, ratio = {}".format(shape, shape_star, ratio_acceptation))
        # print("current_llike = {}, candidate_llike = {}".format(current_llike, candidate_llike))
        # print("current_lprior = {}, candidate_lprior = {}".format(current_lprior, candidate_lprior))
    return [loc, scale, shape]


def medIQR(X):
    return np.round([np.median(X), scipy.stats.iqr(X)], 8)


def IQR_init(N, med, IQR, distribution,epsilon=0):
    if epsilon==0: epsilon=1/N
    loc, scale, shape = 0, 1, 1

    if distribution in ["lognormal", "weibull"]:
        # if MAD > med:
        #     raise Exception("ERROR: MAD > med impossible for {} distribution !".format(distribution))
        n = N // 4
        q1 = med - IQR / 2
        q3 = med + IQR / 2
        if distribution == "lognormal":
            init_theta = [np.log(med), MAD / med, None]
        elif distribution == "weibull":
            init_theta = [0, med / np.log(2), shape]

        if N % 4 == 1:
            X_0 = np.repeat(
                    [
                        q1,
                        q3,
                        med,
                        med - 3 * IQR / 4,
                        med - IQR / 4,
                        med + IQR / 4,
                        med + 3 * IQR / 4,
                    ],
                    [1, 1, 1, n, n - 1, n - 1, n],
                )
            
        elif N % 4 == 3:
            g1, g3 = 1 / 2, 1 / 2
            q1a, q3a = q1 - epsilon * IQR, q3 - epsilon * IQR
            q3b = q3 + epsilon * IQR
            q1b = ((1 - g3) * q3a + g3 * q3b - (1 - g1) * q1a - IQR) / g1
            X_0 = np.repeat(
                [
                    q1a,
                    q1b,
                    q3a,
                    q3b,
                    med,
                    med - 3 * IQR / 4,
                    med - IQR / 4,
                    med + IQR / 4,
                    med + 3 * IQR / 4,
                ],
                [1, 1, 1, 1, 1, n, n - 1, n - 1, n],
            )
        elif N % 2 == 0:
            g1, g3 = 3 / 4, 1 / 4
            q1a, q3a = q1 - epsilon * IQR, q3 - epsilon * IQR
            q3b = q3 + epsilon * IQR
            q1b = ((1 - g3) * q3a + g3 * q3b - (1 - g1) * q1a - IQR) / g1
            med1, med2 = med - epsilon * IQR, med + epsilon * IQR
            X_0 = np.repeat(
                [
                    q1a,
                    q1b,
                    q3a,
                    q3b,
                    med1,
                    med2,
                    med - 3 * IQR / 4,
                    med - IQR / 4,
                    med + IQR / 4,
                    med + 3 * IQR / 4,
                ],
                [1, 1, 1, 1, 1, 1, n - 1, n - 2, n - 2, n - 1],
            )
        else:
            g1, g3 = 1 / 4, 3 / 4
            q1a, q3a = q1 - epsilon * IQR, q3 - epsilon * IQR
            q3b = q3 + epsilon * IQR
            q1b = ((1 - g3) * q3a + g3 * q3b - (1 - g1) * q1a - IQR) / g1
            med1, med2 = med - epsilon * IQR, med + epsilon * IQR
            X_0 = np.repeat(
                [
                    q1a,
                    q1b,
                    q3a,
                    q3b,
                    med1,
                    med2,
                    med - 3 * IQR / 4,
                    med - IQR / 4,
                    med + IQR / 4,
                    med + 3 * IQR / 4,
                ],
                [1, 1, 1, 1, 1, 1, n, n - 2, n - 2, n],
            )
        print(medIQR(X_0))
        X_0 = np.round(np.sort(X_0), 8)

    else:
        if distribution == "normal":
            Z = np.round(np.random.normal(loc, scale, N), 8)
        elif distribution == "cauchy":
            Z = np.round(scipy.stats.cauchy(loc=loc, scale=scale).rvs(N), 8)
        elif distribution == "translated_weibull":
            Z = np.round(weibull_min(c=shape, loc=loc, scale=scale).rvs(N), 8)
        else:
            raise Exception("UKNOWN DISTRIBUTION")
        m_Z, s_Z = np.median(Z), scipy.stats.iqr(Z)
        X_0 = np.sort(np.round((Z- m_Z) / s_Z * IQR + med, 8))
        if distribution == "normal":
            init_theta = [med, IQR / 2 * 1.4826, shape]
        elif distribution == "cauchy":
            init_theta = [med, IQR / 2, shape]
        elif distribution == "translated_weibull":
            init_theta = [
                (loc - m_Z) / s_Z * IQR / 2 + med,
                scale * IQR / 2 / s_Z,
                shape,
            ]

    P = [0.25, 0.5, 0.75]
    H = np.array(P) * (N - 1) + 1
    I = np.floor(H).astype(int)
    G = np.round(H - I, 8)
    Q_tot = []
    K = []

    for k in range(len(I)):
        if G[k] == 0:
            Q_tot.append(X_0[I[k] - 1])
            K.append(I[k])
        else:
            Q_tot.append(X_0[I[k] - 1])
            Q_tot.append(X_0[I[k]])
            K.append(I[k])
            K.append(I[k] + 1)
    if N % 4 == 1:
        Q_sim = [Q_tot[0]]
    elif N % 4 == 3:
        Q_sim = [Q_tot[0], Q_tot[3], Q_tot[4]]
    else:
        Q_sim = [Q_tot[0], Q_tot[2], Q_tot[4], Q_tot[5]]

    K = np.array(K)
    return X_0, init_theta, Q_sim, Q_tot, K, G, I


def X_m_IQR(med, IQR, Q_sim, Q_tot, N, theta, K, G, I, distribution, std_prop):
    loc, scale, shape = theta
    Q_sim, Q_tot = m_IQR_MH(
        med, IQR, Q_sim, Q_tot, N, theta, K, G, I, distribution, std_prop
    )

    K1 = [K[0] - 1] + list(K[1:] - K[:-1] - 1) + [N - K[-1]]

    X1 = np.insert(np.array(Q_tot).astype(float), 0, -np.inf)

    X2 = np.append(Q_tot, np.inf)

    a, b = np.repeat(X1, K1), np.repeat(X2, K1)
    
    sample = truncated(
        a=(a - loc) / scale,
        b=(b - loc) / scale,
        loc=np.repeat(loc, len(a)),
        scale=np.repeat(scale, len(a)),
        size=len(a),
        distribution=distribution,
        shape=shape,
    )
    return np.round(np.append(sample, Q_tot).reshape(-1), 8), Q_sim, Q_tot


def m_IQR_MH(m, IQR, Q_sim, Q_tot, N, theta, K, G, I, distribution, std_prop):
    def log_f_order_stats(X, K, N, theta, distribution):
        loc, scale, shape = theta
        if distribution == "normal":
            f, F = scipy.stats.norm(loc, scale).pdf, scipy.stats.norm(loc, scale).cdf
        elif distribution == "cauchy":
            f, F = (
                scipy.stats.cauchy(loc, scale).pdf,
                scipy.stats.cauchy(loc, scale).cdf,
            )
        elif distribution == "weibull" or distribution == "weibull2":
            f, F = (
                scipy.stats.weibull_min(shape, loc=loc, scale=scale).pdf,
                scipy.stats.weibull_min(shape, loc=loc, scale=scale).cdf,
            )

        X1 = np.insert(np.array(X).astype(float), 0, -np.inf)
        X2 = np.append(X1, np.inf)
        K1 = np.insert(K, 0, 0)
        K1 = np.append(K1, N + 1)
        res = (
            np.sum(np.log(f(X)))
            + (K[0] - 1) * np.log(F(X[0]))
            + (N - K[-1]) * np.log(1 - F(X[-1]))
        )
        for i in range(len(X) - 1):
            res += (K[i + 1] - K[i] - 1) * np.log(F(X[i + 1]) - F(X[i]))
        return res

    loc, scale, shape = theta
    log_density_current = log_f_order_stats(Q_tot, K, N, theta, distribution)
    if distribution == "normal":
        f, Q = norm(loc, scale).pdf, norm(loc, scale).ppf
    elif distribution == "cauchy":
        f, Q = scipy.stats.cauchy(loc, scale).pdf, scipy.stats.cauchy(loc, scale).ppf
    elif distribution[:7] == "weibull":
        f, Q = (
            weibull_min(shape, loc=loc, scale=scale).pdf,
            weibull_min(shape, loc=loc, scale=scale).ppf,
        )

    if N % 4 == 1:
        I_sim = [I[0]]
        Norm = 1 / (1 - G[0])
        Tot_to_sim = [0]
    elif N % 4 == 3:
        I_sim = [I[0], I[2], I[2] + 1]
        Norm = np.array([1 / (1 - G[0]), 1 / G[2] / (1 - G[2]), 1 / (G[2])])
        Tot_to_sim = [0, 2, 3]
    else:
        I_sim = [I[0], I[1], I[2], I[2] + 1]
        Norm = np.array(
            [1 / (1 - G[0]), 1 / (1 - G[1]), 1 / G[2] / (1 - G[2]), 1 / (G[2])]
        )
        Tot_to_sim = [0, 1, 2, 3]
    I_sim = np.array(I_sim)
    p = I_sim / (N + 1)
    Var_K = p * (1 - p) / ((N + 2) * f(Q(p)) ** 2)
    Std_Kernel = np.array(std_prop * np.sqrt(Var_K)) * Norm
    Q_sim_star_full = np.random.normal(Q_sim, Std_Kernel)

    for i in range(len(Q_sim)):
        Q_sim_star = Q_sim.copy()
        Q_sim_star[i] = Q_sim_star_full[i]
        if N % 4 == 1:
            Q_tot_star = [Q_sim_star[0], m, Q_sim_star[0] + IQR]
        elif N % 4 == 3:
            Q_tot_star = [
                Q_sim_star[0],
                (
                    (1 - G[2]) * Q_sim_star[1]
                    + G[2] * Q_sim_star[2]
                    - (1 - G[0]) * Q_sim_star[0]
                    - IQR
                )
                / G[0],
                m,
                Q_sim_star[1],
                Q_sim_star[2],
            ]
        else:
            Q_tot_star = [
                Q_sim_star[0],
                (
                    (1 - G[2]) * Q_sim_star[2]
                    + G[2] * Q_sim_star[3]
                    - (1 - G[0]) * Q_sim_star[0]
                    - IQR
                )
                / G[0],
                Q_sim_star[1],
                2 * m - Q_sim_star[1],
                Q_sim_star[2],
                Q_sim_star[3],
            ]

        if (Q_tot_star == np.sort(Q_tot_star)).all():
            log_density_candidate = log_f_order_stats(
                Q_tot_star, K, N, theta, distribution
            )
            log_density_current = log_f_order_stats(
                Q_tot, K, N, theta, distribution
            )
            ratio = np.exp(log_density_candidate - log_density_current)
            if np.random.uniform(0, 1) < ratio:
                Q_sim[i] = Q_sim_star_full[i]

        if N % 4 == 1:
            Q_tot = [Q_sim[0], m, Q_sim[0] + IQR]
        elif N % 4 == 3:
            Q_tot = [
                Q_sim[0],
                (
                    (1 - G[-1]) * Q_sim[-2]
                    + G[-1] * Q_sim[-1]
                    - (1 - G[0]) * Q_sim[0]
                    - IQR
                )
                / G[0],
                m,
                Q_sim[-2],
                Q_sim[-1],
            ]
        else:
            Q_tot = [
                Q_sim[0],
                (
                    (1 - G[-1]) * Q_sim[-2]
                    + G[-1] * Q_sim[-1]
                    - (1 - G[0]) * Q_sim[0]
                    - IQR
                )
                / G[0],
                Q_sim[1],
                2 * m - Q_sim[1],
                Q_sim[-2],
                Q_sim[-1],
            ]

    return Q_sim, Q_tot


def Gibbs_med_IQR(
    T: int,
    N: int,
    med: float,
    IQR: float,
    distribution: str = "normal",
    prior_loc: str = "normal",
    prior_scale: str = "gamma",
    prior_shape: str = "gamma",
    par_prior_loc: list = [0, 1],
    par_prior_scale: list = [1, 1],
    par_prior_shape: list = [1, 1],
    std_prop_loc: float = 0.1,
    std_prop_scale: float = 0.1,
    std_prop_shape: float = 0.1,
    std_prop_quantile: float = 0.1,
    List_X: bool = False,
    verbose: bool = True,
) -> dict:
    
    """Gibbs sampler to sample from the posterior of model parameters given the median and IQR of the data.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """    
    
    if distribution == "weibull":
        par_names=["scale","shape"]
    elif distribution=="translated_weibull":
        par_names=["loc","scale","shape"]
    elif distribution=="normal" or distribution=="cauchy":
        par_names=["loc","scale"]
    else: raise Exception("ERROR: distribution {} not implemented !".format(distribution))
    
    X_0, init_theta, q_sim, q_tot, K, G, I = IQR_init(N, med, IQR, distribution)

    Theta = [init_theta]
    X_list = [X_0]

    Q_Tot = [q_tot]
    Q_Sim = [q_sim]

    X = X_0.copy()

    for i in tqdm(range(T), disable=not (verbose)):
        X, q_sim, q_tot = X_m_IQR(
            med,
            IQR,
            q_sim,
            q_tot,
            N,
            Theta[-1],
            K,
            G,
            I,
            distribution,
            std_prop_quantile
        )

        theta = posterior(
            X,
            Theta[-1],
            distribution,
            prior_loc,
            prior_scale,
            prior_shape,
            par_prior_loc,
            par_prior_scale,
            par_prior_shape,
            std_prop_loc,
            std_prop_scale,
            std_prop_shape,
        )

        Theta.append(theta)

        Q_Tot.append(list(q_tot))
        Q_Sim.append(list(q_sim))
        if List_X:
            X_list.append(X.copy())

    if not (List_X):
        X_list.append(X.copy())
        
    Q = np.array(Q_Sim).T
    Theta = np.array(Theta).T
    chains0={par_name:Theta[i] for i,par_name in enumerate(["loc","scale","shape"])}
    chains = {par_name: chains0[par_name] for par_name in par_names}
    
    
    if verbose:
        print("Acceptance rates of Quantile :",end=" ")
        for i in range(Q.shape[0]):
            q = Q[i]
            print(
                "Q {} = {:.2%}".format(i, len(np.unique(q)) / len(q))
            )

    if verbose and prior_loc!="NIG":
        acceptation_rate=[(len(np.unique(chains[par_name]))-1)/T for par_name in par_names]
        print('Acceptation rates MH :',end=" ")
        for i in range(len(par_names)):
            print("{} = {:.2%}".format(par_names[i],acceptation_rate[i]),end=" ")
        print()
        
    return {
        "X": X_list,
        "chains": chains,
        "N": N,
        "med": med,
        "IQR": IQR,
        "T": T,
        "distribution": distribution,
        "prior_loc":prior_loc,
        "prior_scale": prior_scale,
        "prior_shape": prior_shape,
        "par_prior_loc": par_prior_loc,
        "par_prior_scale": par_prior_scale,
        "par_prior_shape": par_prior_shape,
        "Q_sim": np.array(Q_Sim),
        "Q_tot": np.array(Q_Tot)
    }
