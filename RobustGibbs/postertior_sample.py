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
    """Function to sample from the posterior of parameters theta given data X. 
    """    
    # Special Normal/NIG case
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
        return [mu, 1 / np.sqrt(tau),0]
    
    if distribution == "normal" and prior_loc == "NIG":
        return post_NIG(X, par_prior_loc, par_prior_scale)

    def llike_cauchy(x, loc, scale, shape):
        return np.sum(np.log(cauchy.pdf(x, loc=loc, scale=scale)))

    def llike_normal(x, loc, scale, shape):
        return np.sum(np.log(norm.pdf(x, loc=loc, scale=scale)))

    def llike_weibull(x, loc, scale, shape):
        return np.sum(np.log(weibull_min.pdf(x, c=shape, scale=scale, loc=loc)))
    
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
            if par_prior_loc[0] <= 0 or par_prior_loc[1]<=0:
                raise Exception(
                    "ERROR : prior location parameter invalid for gamma location prior!"
                )
            lprior_loc = lambda x: np.log(
                gamma(a=par_prior_loc[0], scale=par_prior_loc[1]).pdf(x)
            )
        else:
            raise Exception(
                "ERROR : Prior for location {} not valid !".format(prior_loc)
            )

        loc_star = np.random.normal(loc, std_prop_loc)

        if not((loc_star >= np.min(X) and distribution=="translated_weibull") or (loc_star<=0 and prior_loc == "gamma") or (loc_star<=0 and distribution=="weibull")):
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
        if par_prior_scale[0] <= 0 or par_prior_scale[1]<=0:
            raise Exception(
                    "ERROR : prior location parameter invalid for gamma scale prior!"
                )
        lprior_scale = lambda x: np.log(
            gamma(a=par_prior_scale[0], scale=par_prior_scale[1]).pdf(x)
        )
    elif prior_scale=="jeffreys":
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
            if par_prior_shape[0] <= 0 or par_prior_shape[1]<=0:
                raise Exception(
                    "ERROR : prior location parameter invalid for gamma shape prior!"
                )
            lprior_shape = lambda x: np.log(
                gamma(a=par_prior_shape[0], scale=par_prior_shape[1]).pdf(x)
            )
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
                
    return [loc, scale, shape]
