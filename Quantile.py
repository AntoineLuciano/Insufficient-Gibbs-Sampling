import numpy as np
import scipy.stats
from tqdm import tqdm
from truncated import *

from Quantile import *
from normal_post import *
from cauchy_post import *
from weibull_post import *


def Quantile_Init(Q, P, N, loc, scale, distribution):
    H = np.array(P) * (N - 1) + 1
    I = np.floor(H)
    G = np.round(H - I, 8)

    K = []
    if G[0] == 0:
        delta_2 = 0
    else:
        delta_2 = 1

    X1 = truncated(
        loc,
        scale,
        -np.inf,
        (Q[0] - loc) / scale,
        size=int(I[0] - 1 + delta_2),
        distribution=distribution,
    )

    Q_tot = np.array([])
    K = np.array([])
    if G[0] == 0:
        X1 = np.append(X1, Q[0])
        Q_tot = np.append(Q_tot, Q[0])
        K = np.append(K, I[0])

    Q_sim = np.array([])

    for i in range(len(Q) - 1):
        delta_1 = delta_2

        if delta_1 > 0:
            x1 = np.max(X1)
            Q_sim = np.append(Q_sim, x1)
            x2 = (Q[i] - x1 * (1 - G[i])) / G[i]
            Q_tot = np.append(Q_tot, [x1, x2])
            K = np.append(K, [I[i], I[i] + 1])
            X1 = np.append(X1, x2)
            print("X1 = {} , Q = {} G = {} X2 = {}".format(x1, Q[i], G[i], x2))

        if G[i + 1] > 0:
            delta_2 = 1
        else:
            delta_2 = 0

        if np.max(X1) > Q[i + 1]:
            print("Probleme de simulation des quantiles lors de l'Initialisation")
            return None
        print(np.max(X1), Q[i], Q[i + 1])
        X1 = np.append(
            X1,
            truncated(
                loc,
                scale,
                (max(np.max(X1), Q[i]) - loc) / scale,
                (Q[i + 1] - loc) / scale,
                size=int(I[i + 1] - I[i]) - 1 - delta_1 + delta_2,
                distribution=distribution,
            ),
        )
        if G[i + 1] == 0:
            X1 = np.append(X1, Q[i + 1])
            Q_tot = np.append(Q_tot, Q[i + 1])
            K = np.append(K, I[i + 1])

    if delta_2 > 0:
        x1 = np.max(X1)
        Q_sim = np.append(Q_sim, x1)
        x2 = (Q[-1] - x1 * (1 - G[-1])) / G[-1]
        Q_tot = np.append(Q_tot, [x1, x2])
        K = np.append(K, [I[-1], I[-1] + 1])
        X1 = np.append(X1, x2)
    X1 = np.append(
        X1,
        truncated(
            loc,
            scale,
            (max(Q[-1], np.max(X1)) - loc) / scale,
            np.inf,
            size=(N - int(I[-1]) - delta_2),
            distribution=distribution,
        ),
    )

    return X1, I, G, Q_sim, Q_tot, K


def Quantile_Init2(Q, P, N, loc, scale, distribution, epsilon=0.001, shape=1):
    H = np.array(P) * (N - 1) + 1
    I = np.floor(H)
    G = np.round(H - I, 8)
    #print("Dans quantile init2, I = ",len(I))
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
                    print(
                        "Probleme de simulation des quantiles lors de l'Initialisation2",
                        Q_tot[-1],
                        Q[k + 1],
                    )
                    return None
    # print(Q_tot)
    K = np.array(K)
    K1 = [K[0] - 1] + list(K[1:] - K[:-1] - 1) + [N - K[-1]]
    # print(K1,sum(K1),len(Q_tot))
    X1 = np.insert(np.array(Q_tot).astype(float), 0, -np.inf)

    X2 = np.append(Q_tot, np.inf)
    a, b = np.repeat(X1, K1), np.repeat(X2, K1)

    # print(np.array([a,b]).T)
    sample = truncated(
        a=(a - loc) / scale,
        b=(b - loc) / scale,
        loc=np.repeat(loc, len(a)),
        scale=np.repeat(scale, len(a)),
        size=len(a),
        distribution=distribution,
        shape=shape,
    )
    return np.round(np.append(sample, Q_tot).reshape(-1), 8), I, G, Q_sim, Q_tot, K


def pdf_lognorm(x, loc, scale):
    return scipy.stats.norm(loc, scale).pdf(np.log(x)) / x


def log_f_order_stats(X, K, N, loc, scale, distribution, shape=1):
    if distribution == "normal":
        f, F = scipy.stats.norm(loc, scale).pdf, scipy.stats.norm(loc, scale).cdf

    # elif distribution=="lognormal":
    #     F=lambda x: scipy.stats.norm(loc,scale).cdf(np.log(x))
    #     f=lambda x: pdf_lognorm(x,loc,scale)
    elif distribution == "cauchy":
        f, F = scipy.stats.cauchy(loc, scale).pdf, scipy.stats.cauchy(loc, scale).cdf
    elif distribution == "weibull" or distribution == "weibull2":
        f, F = (
            scipy.stats.weibull_min(shape, loc=loc, scale=scale).pdf,
            scipy.stats.weibull_min(shape, loc=loc, scale=scale).cdf,
        )

    X1 = np.insert(np.array(X).astype(float), 0, -np.inf)

    X2 = np.append(X1, np.inf)
    K1 = np.insert(K, 0, 0)
    K1 = np.append(K1, N + 1)

    # print(K1[1]-K1[0]-1,F(X2[1]),F(X2[0]),X2[0],np.log(F(X2[1])-F(X2[0])))
    # print(len(K[1:]),len(K[:-1]),len(X2[1:]),len(X2[:-1]),len(F(X2[1:])))
    # print(np.sum(np.log(f(X))),(K[1:]-K[:-1]-1)*np.log(F(X2[1:])-F(X2[:-1]))[0],(K[1]-K[0]-1),np.log(F(X2[1])-F(X2[0])),np.sum((K[1:]-K[:-1]-1)*np.log(F(X2[1:])-F(X2[:-1]))))

    # res2 = np.sum(np.log(f(X)))+np.sum((K[1:]-K[:-1]-1)*np.log(F(X2[1:])-F(X2[:-1])))

    res = (
        np.sum(np.log(f(X)))
        + (K[0] - 1) * np.log(F(X[0]))
        + (N - K[-1]) * np.log(1 - F(X[-1]))
    )

    for i in range(len(X) - 1):
        res += (K[i + 1] - K[i] - 1) * np.log(F(X[i + 1]) - F(X[i]))

    # print(res,res2,res==res2)
    return res


def Quantile_MH(
    Q_val, Q_sim, Q_tot, N, loc, scale, K, I, G, distribution, std_prop, shape=1
):
    if distribution == "normal":
        f, Q = scipy.stats.norm(loc, scale).pdf, scipy.stats.norm(loc, scale).ppf

    # elif distribution=="lognormal":
    #     f,Q=scipy.stats.norm(loc,scale).pdf,scipy.stats.norm(loc,scale).ppf
    #     Q_val,Q_sim,Q_tot=np.log(Q_val),np.log(Q_sim),np.log(Q_tot)
    # distribution="normal"

    # Q= lambda x: np.exp(scipy.stats.norm(loc,scale).ppf(x))
    # f=lambda x: pdf_lognorm(x,loc,scale)
    elif distribution == "cauchy":
        f, Q = scipy.stats.cauchy(loc, scale).pdf, scipy.stats.cauchy(loc, scale).ppf

    elif distribution == "weibull" or distribution == "weibull2":
        if distribution == "weibull2":
            loc = 0
        f, Q = (
            scipy.stats.weibull_min(shape, loc=loc, scale=scale).pdf,
            scipy.stats.weibull_min(shape, loc=loc, scale=scale).ppf,
        )

    I_sim = np.array(I[np.where(G > 0)])
    p = I_sim / (N + 1)
    #print(len(p), len(N), len(Q(p)))
    Var_K = p * (1 - p) / ((N + 2) * f(Q(p)) ** 2)

    Std_Kernel = std_prop * np.sqrt(Var_K) / (1 - G[np.where(G > 0)])
    Q_sim_star = np.random.normal(Q_sim[:len(Std_Kernel)], Std_Kernel)

    log_density_current = log_f_order_stats(
        Q_tot, K, N, loc, scale, distribution, shape=shape
    )

    Q_tot_star = []

    j = 0
    for i in range(len(Q_val)):
        if G[i] > 0:
            Q_tot_star.append(Q_sim_star[j])
            Q_tot_star.append((Q_val[i] - Q_sim_star[j] * (1 - G[i])) / G[i])
            j += 1
        else:
            Q_tot_star.append(Q_val[i])

    if (Q_tot_star != np.sort(Q_tot_star)).any():
        return Q_sim, Q_tot
    log_density_candidate = log_f_order_stats(
        Q_tot_star, K, N, loc, scale, distribution, shape=shape
    )
    ratio = np.exp(log_density_candidate - log_density_current)
    if np.random.uniform(0, 1) < ratio:
        return Q_sim_star, Q_tot_star
    return Q_sim, Q_tot


def X_Q(Q_val, Q_sim, Q_tot, N, loc, scale, K, I, G, distribution, std_prop, shape=1):
    if len(Q_sim) > 0:
        Q_sim, Q_tot = Quantile_MH_test(
            Q_val,
            Q_sim,
            Q_tot,
            N,
            loc,
            scale,
            K,
            I,
            G,
            distribution,
            std_prop,
            shape=shape,
        )
    # print(Q_tot,Q_val)
    K1 = [K[0] - 1] + list(K[1:] - K[:-1] - 1) + [N - K[-1]]
    # print(Q_sim,Q_tot)
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


def Gibbs_Quantile(
    T,
    N,
    Q,
    P,
    n_chains,
    distribution,
    par_prior=[0, 1, 1, 1, 1, 1],
    std_prop1=0.1,
    std_prop2=0.1,
    std_prop3=0.1,
    std_prop_quantile=0.1,
    List_X=False,
    epsilon=0.001,
    verbose=True,
    shape=1,
):
    log_norm = False
    if distribution == "normal":
        loc = Q[len(Q) // 2]

        scale = (Q[-1] - Q[0]) / (
            scipy.stats.norm(loc).ppf(P[-1]) - scipy.stats.norm(loc).ppf(P[0])
        )
        theta = [loc, scale]
        if verbose:
            print("Init loc = {} and scale = {}".format(loc, scale))
    if distribution == "lognormal":
        Q = np.log(Q)
        loc = Q[len(Q) // 2]
        scale = (Q[-1] - Q[0]) / (
            scipy.stats.norm(loc).ppf(P[-1]) - scipy.stats.norm(loc).ppf(P[0])
        )
        theta = [loc, scale]
        log_norm = True
        distribution = "normal"
        if verbose:
            print("Init loc = {} and scale = {}".format(loc, scale))
    if distribution == "cauchy":
        loc = Q[len(Q) // 2]
        scale = (Q[-1] - Q[0]) / (
            scipy.stats.cauchy(loc).ppf(P[-1]) - scipy.stats.cauchy(loc).ppf(P[0])
        )
        theta = [loc, scale]

        if verbose:
            print("Init loc = {} and scale = {}".format(loc, scale))
    if distribution == "weibull" or distribution == "weibull2":
        loc = Q[0] - (Q[1] - Q[0])
        if distribution == "weibull2":
            loc = 0
        shape = 1.5
        scale = (Q[-1] - Q[0]) / (
            scipy.stats.weibull_min(shape, loc=loc).ppf(P[-1])
            - scipy.stats.weibull_min(shape, loc=loc).ppf(P[0])
        )
        # loc,shape,scale=10,3,2
        theta = [loc, scale, shape]

        if verbose:
            print("Init loc = {},scale = {} shape = {}".format(loc, scale, shape))
    X, I, G, q_sim, q_tot, K = Quantile_Init2(
        Q, P, N, loc, scale, distribution, epsilon=epsilon, shape=shape
    )
    # print("Min = {}, Max = {}".format(np.min(X),np.max(X)))
    # print(X)

    Theta = [theta]
    X_list = [X]
    Mean = [np.mean(X)]
    Std = [np.std(X)]
    Q_Tot = [q_tot]
    Q_Sim = [q_sim]

    for i in tqdm(range(T), disable=not (verbose)):
        X, q_sim, q_tot = X_Q(
            Q,
            q_sim,
            q_tot,
            N,
            Theta[-1][0],
            Theta[-1][1],
            K,
            I,
            G,
            distribution,
            std_prop_quantile,
            shape=shape,
        )

        if distribution == "normal":
            mu, tau = post_NG(X, par_prior)
            theta = [mu, 1 / np.sqrt(tau)]
        elif distribution == "lognormal":
            mu, tau = post_NG(np.log(X), par_prior)
            # print(np.std(np.log(X)),np.sqrt(1/tau))
            theta = [mu, np.sqrt(1 / tau)]
        elif distribution == "cauchy":
            loc = post_cauchy_theta(
                Theta[-1][0], Theta[-1][1], X, par_prior[:2], std_prop1
            )
            scale = post_cauchy_gamma(loc, Theta[-1][1], X, par_prior[2:], std_prop2)
            theta = [loc, scale]

        elif distribution == "weibull" or distribution == "weibull2":
            # print(np.min(X),loc)
            if distribution == "weibull2":
                loc = 0
            else:
                loc = post_weibull_loc2(
                    Theta[-1][0],
                    Theta[-1][1],
                    Theta[-1][2],
                    X,
                    par_prior[:2],
                    std_prop1,
                )
            scale = post_weibull_scale(
                loc, Theta[-1][1], Theta[-1][2], X, par_prior[2:4], std_prop2
            )
            shape = post_weibull_k(
                loc, scale, Theta[-1][2], X, par_prior[4:], std_prop3
            )
            theta = [loc, scale, shape]

        Theta.append(theta)
        Mean.append(np.mean(X))
        Std.append(np.std(X))

        Q_Tot.append(list(q_tot))

        Q_Sim.append(list(q_sim))
        if List_X:
            X_list.append(X)

    if not (List_X):
        X_list.append(X)

    if log_norm:
        X_list = np.exp(np.array(X_list))
        Q_Sim = np.exp(np.array(Q_Sim))
        Q_Tot = np.exp(np.array(Q_Tot))

    if verbose:
        Q = np.array(Q_Sim).T
        print("I=", I, "Q.shape=", Q.shape)
        for i in range(Q.shape[0]):
            q = Q[i]
            print(
                "Acceptance rate of Q {} = {:.2%}".format(i, len(np.unique(q)) / len(q))
            )
    # if verbose:print("Acceptance rate of Q =  {:.2%}".format((len(np.unique(Q_Sim,axis=0))-1)/len(Q_Sim)))
    if verbose and distribution == "cauchy":
        print(
            "Acceptation rate of loc = {:.2%} and of scale = {:.2%}".format(
                len(np.unique(np.array(Theta)[:, 0], axis=0)) / len(Theta),
                len(np.unique(np.array(Theta)[:, 1], axis=0)) / len(Theta),
            )
        )
    if verbose and (distribution == "weibull" or distribution == "weibull2"):
        print(
            "Acceptation rate of loc = {:.2%}, of scale = {:.2%} and of shape = {:.2%}".format(
                len(np.unique(np.array(Theta)[:, 0], axis=0)) / len(Theta),
                len(np.unique(np.array(Theta)[:, 1], axis=0)) / len(Theta),
                len(np.unique(np.array(Theta)[:, 2], axis=0)) / len(Theta),
            )
        )
    return {
        "X": X_list,
        "Mean": Mean,
        "Std": Std,
        "chains": np.array(Theta).T,
        "Q_sim": np.array(Q_Sim),
        "Q_tot": np.array(Q_Tot),
        "I": I,
        "K": K,
        "G": G,
        "par_prior": par_prior,
        "distribution": distribution,
        "Q": Q,
        "P": P,
        "N": N,
    }


def log_density(X, I, loc, scale, distribution, shape=1):
    if distribution == "normal":
        f, F = scipy.stats.norm(loc, scale).pdf, scipy.stats.norm(loc, scale).cdf

    # elif distribution=="lognormal":
    #     F=lambda x: scipy.stats.norm(loc,scale).cdf(np.log(x))
    #     f=lambda x: pdf_lognorm(x,loc,scale)
    elif distribution == "cauchy":
        f, F = scipy.stats.cauchy(loc, scale).pdf, scipy.stats.cauchy(loc, scale).cdf
    elif distribution == "weibull" or distribution == "weibull2":
        f, F = (
            scipy.stats.weibull_min(shape, loc=loc, scale=scale).pdf,
            scipy.stats.weibull_min(shape, loc=loc, scale=scale).cdf,
        )

    return (
        np.log(F([X[1]]) - F(X[0])) * (I[1] - I[0] - 1)
        + np.log(F([X[3]]) - F(X[2])) * (I[3] - I[2] - 1)
        + np.log(f(X[2]))
        + np.log(f(X[1]))
    )


def Quantile_MH_test(
    Q_val, Q_sim, Q_tot, N, loc, scale, K, I, G, distribution, std_prop, shape=1
):
    if distribution == "normal":
        f, Q = scipy.stats.norm(loc, scale).pdf, scipy.stats.norm(loc, scale).ppf

    # elif distribution=="lognormal":
    #     f,Q=scipy.stats.norm(loc,scale).pdf,scipy.stats.norm(loc,scale).ppf
    #     Q_val,Q_sim,Q_tot=np.log(Q_val),np.log(Q_sim),np.log(Q_tot)
    # distribution="normal"

    # Q= lambda x: np.exp(scipy.stats.norm(loc,scale).ppf(x))
    # f=lambda x: pdf_lognorm(x,loc,scale)
    elif distribution == "cauchy":
        f, Q = scipy.stats.cauchy(loc, scale).pdf, scipy.stats.cauchy(loc, scale).ppf

    elif distribution == "weibull" or distribution == "weibull2":
        f, Q = (
            scipy.stats.weibull_min(shape, loc=loc, scale=scale).pdf,
            scipy.stats.weibull_min(shape, loc=loc, scale=scale).ppf,
        )

    I_sim = np.array(I[np.where(G > 0)])
    p = I_sim / (N + 1)
    Var_K = p * (1 - p) / ((N + 2) * f(Q(p)) ** 2)

    Std_Kernel = std_prop * np.sqrt(Var_K) / (1 - G[np.where(G > 0)])
    Q_sim_star = np.random.normal(Q_sim[:len(Std_Kernel)], Std_Kernel)

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
    # print(G)
    # print(Q_tot_star)
    # print(len(Q_tot2),len(Q_tot),len(Q_sim))
    #print("len(Q_tot)={}, len(Q_sim)={}, len(Q_tot_star)={}".format(len(Q_tot),len(Q_sim),len(Q_tot_star)))
    while j < len(Q_tot2) - 1:
        # print(i,j,k)
        if k>=len(G): print("ERREUR : k = ",k," len(G) = ",len(G),"len(Q_sim*)=",len(Q_sim_star))
        if G[k] > 0:
            if Q_sim_star[i] < Q_tot2[j - 1] or Q_sim_star[i] > Q_val[k]:
                #print("Quantile simulé {} : Rejeté".format(i))
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
                #print("Quantile simulé {} : Accepté {} {}".format(i,Q_tot_star2[j],Q_tot_star2[j+1]))
                Q_tot[j - 1] = Q_tot_star2[j]
                Q_tot[j] = Q_tot_star2[j + 1]
                Q_sim[i] = Q_tot_star2[j]
            #else: print("Quantile simulé {} : Rejeté".format(i))
            j += 2
            i += 1
        else:
            j += 1
        k += 1
    # print(Q_tot,Q_sim)

    return Q_sim, Q_tot
