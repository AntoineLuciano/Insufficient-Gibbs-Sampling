import numpy as np
import scipy

def cauchy(x, loc, scale):
    return 1 / (np.math.pi * scale * (1 + ((x - loc) / scale) ** 2))


def post_cauchy_theta(theta, gamma, X, par_prior, std_prop,):
    
    def log_prior_theta(theta,par_prior):
        theta_0, gamma_0 = par_prior[0], par_prior[1]
        return - np.log(1+((theta-theta_0)/gamma_0)**2)
    
    def log_like_theta(x,theta,gamma):
        return - np.sum(np.log(1+((x-theta)/gamma)**2))

    theta_0, gamma_0 = par_prior[0], par_prior[1]

    theta_star = np.random.normal(loc=theta, scale=std_prop)
    
    current_llikelihood = log_like_theta(X,theta,gamma)
    
    candidate_llikelihood = log_like_theta(X,theta_star,gamma)
    
    current_lprior = log_prior_theta(theta,par_prior)
    candidate_lprior = log_prior_theta(theta_star,par_prior)
    ratio_acceptation = min(np.exp(candidate_llikelihood - current_llikelihood + candidate_lprior - current_lprior),1)
    if np.random.uniform() < ratio_acceptation:
        return theta_star
    return theta



def log_gamma_pdf(x,shape,scale):
    return -shape*np.log(scale)-scipy.special.gammaln(shape)+(shape-1)*np.log(x)-x/scale


def post_cauchy_gamma(theta, gamma, X, par_prior,std_prop,log=-np.inf):
    
    def log_cauchy_gamma(x, loc, scale):
        return -np.log(scale) - np.log(1 + ((x - loc) / scale) ** 2)
    
    def log_prior_gamma(gamma,par_prior):
        theta_0, gamma_0 = par_prior[0], par_prior[1]
        return - np.log(1+((gamma-gamma_0)/gamma_0)**2)
    
    # def log_prior_gamma(gamma,par_prior):
    #     alpha,beta=par_prior
    #     return (alpha-1)*np.log(gamma)-beta*gamma
    # kernel=scipy.stats.truncnorm(loc=0, scale=std_prop, a=gamma / std_prop, b=np.inf)
    gamma_star = np.random.normal(gamma, std_prop)
    if gamma_star <= 0:
        return gamma
    #X=np.sort(X)[2:-2]
    # current_llikelihood = np.sum(np.log(cauchy(X, theta, gamma)))
    # candidate_llikelihood = np.sum(np.log(cauchy(X, theta, gamma_star)))
    current_llikelihood = np.sum(log_cauchy_gamma(X, theta, gamma))
    candidate_llikelihood = np.sum(log_cauchy_gamma(X, theta, gamma_star))

    #current_lprior = log_prior_gamma(gamma,par_prior)
    #candidate_lprior = log_prior_gamma(gamma_star,par_prior)
    # current_lprior = np.log(1 / gamma)
    # candidate_lprior = np.log(1 / gamma_star)
    
    current_lprior=log_gamma_pdf(gamma,par_prior[0],par_prior[1])
    candidate_lprior=log_gamma_pdf(gamma_star,par_prior[0],par_prior[1])
    # print("Gamma* =",gamma_star,end="")

    # current_kernel = np.log(kernel.pdf(gamma-gamma_star))

    # candidate_kernel = np.log(kernel.pdf(gamma_star-gamma))
    # print("Prior impact = {}, Likelihood impact = {}".format(candidate_lprior - current_lprior,candidate_llikelihood - current_llikelihood))
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
        return gamma_star
    return gamma