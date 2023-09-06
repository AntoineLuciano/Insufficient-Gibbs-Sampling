import numpy as np
from scipy.stats import weibull_min, norm, cauchy, lognorm
from tqdm import tqdm


from RobustGibbs.truncated import *
from RobustGibbs.postertior_sample import posterior

### INITIALISATION


def Quantile_Init(Q, P, N, distribution, epsilon=0.001):
    loc, scale, shape = 0, 1, 1
    if distribution == "normal":
        loc = Q[len(Q) // 2]
        scale = (Q[-1] - Q[0]) / (norm(loc).ppf(P[-1]) - norm(loc).ppf(P[0]))
        par_names = ["loc", "scale"]
    if distribution == "cauchy":
        loc = Q[len(Q) // 2]
        scale = (Q[-1] - Q[0]) / (cauchy(loc).ppf(P[-1]) - cauchy(loc).ppf(P[0]))
        par_names = ["loc", "scale"]

    if distribution == "translated_weibull" or distribution == "weibull":
        if distribution == "weibull":
            loc = 0
            par_names = ["scale", "shape"]

        else:
            loc = 2 * Q[0] - Q[1]
            par_names = ["loc", "scale", "shape"]

        shape = 1.5
        scale = (Q[-1] - Q[0]) / (
            weibull_min(shape, loc=loc).ppf(P[-1])
            - weibull_min(shape, loc=loc).ppf(P[0])
        )

    init_theta = [loc, scale, shape]
    H = np.array(P) * (N - 1) + 1
    I = np.floor(H)
    G = np.round(H - I, 8)
    Q_sim = []
    Q_tot = []
    K = []
    for k in range(len(G)):
        K.append(I[k])
        if G[k] == 0:
            Q_tot.append(Q[k])
        else:
            Q_sim.append(Q[k] - epsilon)
            Q_tot.append(Q[k] - epsilon)
            Q_tot.append((Q[k] - Q_tot[-1] * (1 - G[k])) / G[k])
            K.append(I[k] + 1)
            if k < len(G) - 1:
                if Q_tot[-1] > Q[k + 1]:
                    raise Exception("Initialization problem !")
    K = np.array(K)
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

    X_0 = np.round(np.append(sample, Q_tot).reshape(-1), 8)
    return X_0, init_theta, par_names, Q_sim, Q_tot, K, I, G


### RESAMPLING


def OrderStats_MH(Q_val, Q_sim, Q_tot, N, theta, K, I, G, distribution, std_prop):
    def log_density(X, I, loc, scale, distribution, shape=1):
        if distribution == "normal":
            f, F = norm(loc, scale).pdf, norm(loc, scale).cdf
        elif distribution == "cauchy":
            f, F = cauchy(loc, scale).pdf, cauchy(loc, scale).cdf
        elif distribution == "translated_weibull" or distribution == "weibull":
            f, F = (
                weibull_min(shape, loc=loc, scale=scale).pdf,
                weibull_min(shape, loc=loc, scale=scale).cdf,
            )

        return (
            np.log(F([X[1]]) - F(X[0])) * (I[1] - I[0] - 1)
            + np.log(F([X[3]]) - F(X[2])) * (I[3] - I[2] - 1)
            + np.log(f(X[2]))
            + np.log(f(X[1]))
        )

    loc, scale, shape = theta
    if distribution == "normal":
        f, Q = norm(loc, scale).pdf, norm(loc, scale).ppf
    elif distribution == "cauchy":
        f, Q = cauchy(loc, scale).pdf, cauchy(loc, scale).ppf

    elif distribution == "translated_weibull" or distribution == "translated_weibull":
        f, Q = (
            weibull_min(shape, loc=loc, scale=scale).pdf,
            weibull_min(shape, loc=loc, scale=scale).ppf,
        )

    I_sim = np.array(I[np.where(G > 0)])
    p = I_sim / (N + 1)
    Var_K = p * (1 - p) / ((N + 2) * f(Q(p)) ** 2)

    Std_Kernel = std_prop * np.sqrt(Var_K) / (1 - G[np.where(G > 0)])
    Q_sim_star = np.random.normal(Q_sim[: len(Std_Kernel)], Std_Kernel)

    Q_tot_star = []

    j = 0
    for i in range(len(Q_val)):
        if G[i] > 0:
            Q_tot_star.append(Q_sim_star[j])
            Q_tot_star.append((Q_val[i] - Q_sim_star[j] * (1 - G[i])) / G[i])
            j += 1
        else:
            Q_tot_star.append(Q_val[i])

    Q_tot_star2 = np.array(Q_tot_star)
    Q_tot_star2 = np.insert(Q_tot_star2, 0, -np.inf)
    Q_tot_star2 = np.append(Q_tot_star2, np.inf)
    Q_tot2 = np.array(Q_tot)
    Q_tot2 = np.insert(Q_tot2, 0, -np.inf)
    Q_tot2 = np.append(Q_tot2, np.inf)
    K1 = np.array(K)
    K1 = np.insert(K1, 0, 0)
    K1 = np.append(K1, N + 1)
    i = 0
    j = 1
    k = 0

    while j < len(Q_tot2) - 1:
        if k >= len(G):
            print(
                "ERREUR : k = ",
                k,
                " len(G) = ",
                len(G),
                "len(Q_sim*)=",
                len(Q_sim_star),
            )
        if G[k] > 0:
            if Q_sim_star[i] < Q_tot2[j - 1] or Q_sim_star[i] > Q_val[k]:
                j += 2
                i += 1
                k += 1
                continue

            X_current = Q_tot2[j - 1 : j + 3]
            X_candidate = [
                Q_tot2[j - 1],
                Q_tot_star2[j],
                Q_tot_star2[j + 1],
                Q_tot2[j + 2],
            ]
            I_i = K1[j - 1 : j + 3]
            log_density_current = log_density(
                X_current, I_i, loc, scale, distribution, shape=shape
            )
            log_density_candidate = log_density(
                X_candidate, I_i, loc, scale, distribution, shape=shape
            )
            ratio = np.exp(log_density_candidate - log_density_current)
            if np.random.uniform(0, 1) < ratio:
                Q_tot[j - 1] = Q_tot_star2[j]
                Q_tot[j] = Q_tot_star2[j + 1]
                Q_sim[i] = Q_tot_star2[j]
            j += 2
            i += 1
        else:
            j += 1
        k += 1

    return Q_sim, Q_tot


def Resample_X_Q(
    Q_val, Q_sim, Q_tot, N, theta, K, I, G, distribution, std_prop, shape=1
):
    if len(Q_sim) > 0:
        Q_sim, Q_tot = OrderStats_MH(
            Q_val,
            Q_sim,
            Q_tot,
            N,
            theta,
            K,
            I,
            G,
            distribution,
            std_prop,
        )
    loc, scale, shape = theta
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


### GIBBS SAMPLER


def Gibbs_Quantile(
    T: int,
    N: int,
    Q: list,
    P: list,
    distribution: str = "normal",
    prior_loc: str = "normal",
    prior_scale: str = "gamma",
    prior_shape: str = "gamma",
    par_prior_loc: list = [0, 1],
    par_prior_scale: list = [0, 1],
    par_prior_shape: list = [0, 1],
    std_prop_loc: float = 0.1,
    std_prop_scale: float = 0.1,
    std_prop_shape: float = 0.1,
    std_prop_quantile=0.1,
    List_X=False,
    epsilon=0.001,
    verbose=True,
):
    """Gibbs sampler to sample from the posterior of model parameters given a sequence of quantiles.

       Args:
        T (int): Number of iterations.
        N (int): Size of the vector X.
        Q (list): Observed quantile value.
        P (float): Probability associated to the observed quantiles Q.
        distribution (str): Distribution of the data ("normal", "cauchy", "weibull", or "translated_weibull").
        prior_loc (str): Prior distribution of the location parameter ("normal", "cauchy", "uniform", or "none").
        prior_scale (str): Prior distribution of the scale parameter ("gamma","jeffreys").
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
            ... input parameters"""

    X, init_theta, par_names, q_sim, q_tot, K, I, G = Quantile_Init(
        Q, P, N, distribution, epsilon=epsilon
    )

    Theta = [init_theta]
    X_list = [X]
    Q_Tot = [q_tot]
    Q_Sim = [q_sim]

    for i in tqdm(range(T), disable=not (verbose)):
        X, q_sim, q_tot = Resample_X_Q(
            Q,
            q_sim,
            q_tot,
            N,
            Theta[-1],
            K,
            I,
            G,
            distribution,
            std_prop_quantile,
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
            X_list.append(X)

    if not (List_X):
        X_list.append(X)

    if verbose:
        Q = np.array(Q_Sim).T
        print("I=", I, "Q.shape=", Q.shape)
        for i in range(Q.shape[0]):
            q = Q[i]
            print(
                "Acceptance rate of Q {} = {:.2%}".format(
                    i, (len(np.unique(q)) - 1) / len(q)
                )
            )
    Theta = np.array(Theta).T
    chains0 = {
        par_name: Theta[i] for i, par_name in enumerate(["loc", "scale", "shape"])
    }
    chains = {par_name: chains0[par_name] for par_name in par_names}
    if verbose and prior_loc != "NIG":
        acceptation_rate = [
            (len(np.unique(chains[par_name])) - 1) / T for par_name in par_names
        ]
        print("Acceptation rates MH :", end=" ")
        for i in range(len(par_names)):
            print("{} = {:.2%}".format(par_names[i], acceptation_rate[i]), end=" ")
        print()
    return {
        "X": X_list,
        "chains": chains,
        "N": N,
        "Q": Q,
        "P": P,
        "distribution": distribution,
        "prior_loc": prior_loc,
        "prior_scale": prior_scale,
        "prior_shape": prior_shape,
        "par_prior_loc": par_prior_loc,
        "par_prior_scale": par_prior_scale,
        "par_prior_shape": par_prior_shape,
        "Q_sim": np.array(Q_Sim),
        "Q_tot": np.array(Q_Tot),
        "T": T,
    }
