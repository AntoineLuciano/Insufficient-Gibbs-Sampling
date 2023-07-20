import numpy as np
import scipy

def post_NG(X, par_prior):
    n = len(X)
    [mu_0, nu, alpha, beta] = par_prior[:4]
    B=beta+ np.sum((X - np.mean(X)) ** 2) / 2+ n * nu * (np.mean(X) - mu_0) ** 2 / (2 * (nu + n))
    A=alpha + n / 2
    M=(nu * mu_0 + np.sum(X)) / (nu + n)
    C=nu + n
    #print(B/(A-1))

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
    return [mu, tau]


def post_NIG(X, par_prior):
    n = len(X)
    [mu_0, nu, alpha, beta] = par_prior
    sigma2 = scipy.stats.invgamma(
        a=alpha + n / 2,
        scale=(
            beta
            + np.sum((X - np.mean(X)) ** 2) / 2
            + n * nu * (np.mean(X) - mu_0) ** 2 / (2 * (nu + n))
        ),
    ).rvs(1)[0]
    mu = np.random.normal(
        loc=(nu * mu_0 + np.sum(X)) / (nu + n), scale=np.sqrt(sigma2 / (nu + n)), size=1
    )[0]
    return [mu, sigma2]
