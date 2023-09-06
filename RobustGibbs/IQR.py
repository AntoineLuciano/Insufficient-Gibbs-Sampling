import numpy as np
import scipy
from scipy.stats import norm, cauchy, weibull_min,iqr
from tqdm import tqdm

from truncated import *
from posterior_sample import posterior


def medIQR(X):
    return np.round([np.median(X), iqr(X)], 8)

# INITIALISATION 

def IQR_init(N, med, IQR, distribution,epsilon=0):
    if epsilon==0: epsilon=1/N
    loc, scale, shape = 0, 1, 1

    if distribution in ["lognormal", "weibull"]:
        if IQR > med: raise Exception("ERROR: IQR > med impossible for {} distribution !".format(distribution))
        n = N // 4
        q1 = med - IQR / 2
        q3 = med + IQR / 2
        if distribution == "lognormal":
            init_theta = [np.log(med), IQR / med, None]
            par_names=["loc","scale"]
        elif distribution == "weibull":
            init_theta = [0, med / np.log(2), shape]
            par_names=["scale","shape"]

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
            par_names=["loc","scale"]
        elif distribution == "cauchy":
            Z = np.round(cauchy(loc=loc, scale=scale).rvs(N), 8)
            par_names=["loc","scale"]
        elif distribution == "translated_weibull":
            Z = np.round(weibull_min(c=shape, loc=loc, scale=scale).rvs(N), 8)
            par_names=["loc","scale","shape"]
        else:
            raise Exception("")
        m_Z, s_Z = medIQR(Z)
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
    return X_0, init_theta,par_names, Q_sim, Q_tot, K, G, I

# RESAMPLING 

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

# METROPOLIS-HASTINGS STEP

def m_IQR_MH(m, IQR, Q_sim, Q_tot, N, theta, K, G, I, distribution, std_prop):
    def log_f_order_stats(X, K, N, theta, distribution):
        loc, scale, shape = theta
        if distribution == "normal":
            f, F = norm(loc, scale).pdf, norm(loc, scale).cdf
        elif distribution == "cauchy":
            f, F = (
                cauchy(loc, scale).pdf,
                cauchy(loc, scale).cdf,
            )
        elif distribution == "weibull" or distribution == "weibull2":
            f, F = (
                weibull_min(shape, loc=loc, scale=scale).pdf,
                weibull_min(shape, loc=loc, scale=scale).cdf,
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
        f, Q = cauchy(loc, scale).pdf, cauchy(loc, scale).ppf
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

# GIBBS SAMPLER

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
    
    """Gibbs sampler for sampling from the posterior distribution of model parameters given the median and IQR of the data.

    Args:
        T (int): Number of iterations.
        N (int): Size of the vector X. 
        med (float): Observed median.
        IQR (float): Observed IQR (Interquartile Range).
        distribution (str): Distribution of the data ("normal", "cauchy", "weibull", or "translated_weibull").
        prior_loc (str): Prior distribution of the location parameter ("normal", "cauchy", "uniform", or "none").
        prior_scale (str): Prior distribution of the scale parameter ("gamma","jeffreys")
        prior_shape (str): Prior distribution of the shape parameter ("gamma").
        par_prior_loc (list, optional): Prior hyperparameters for the location parameter. Defaults to [0, 1].
        par_prior_scale (list, optional): Prior hyperparameters for the scale parameter. Defaults to [1, 1].
        par_prior_shape (list, optional): Prior hyperparameters for the shape parameter. Defaults to [0, 1].
        std_prop_loc (float, optional): Standard deviation of the RWMH Kernel for the location parameter. Defaults to 0.1.
        std_prop_scale (float, optional): Standard deviation of the RWMH Kernel for the scale parameter. Defaults to 0.1.
        std_prop_shape (float, optional): Standard deviation of the RWMH Kernel for the shape parameter. Defaults to 0.1.
        List_X (bool, optional): If True, will return the list of all latent vectors X. Otherwise, it will return the first and the last. Defaults to False.
        verbose (bool, optional): If True, will display the progression of the sampling. Defaults to True.

    Returns:
        A dictionary containing:
            chains (dict): The chains sampled from the parameters' posterior.
            X (list): List of latent vectors. 
            Q_sim (list): List of all simulated order statistics at each iteration. 
            Q_tot (list): List of all order statistics considered at each iteration.
            ... input parameters
        """ 

    

    
    X_0, init_theta, par_names,q_sim, q_tot, K, G, I = IQR_init(N, med, IQR, distribution)

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
