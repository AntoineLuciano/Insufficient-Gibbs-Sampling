import numpy as np
from scipy.stats import norm,truncnorm,weibull_min,cauchy


def truncated(loc, scale, shape=1, a=-np.inf, b=np.inf, distribution="normal", size=1):
    """
    Function to sample from truncated distribution in interval [a,b].
    """
    if distribution == "normal":
        res=truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=size)
    elif distribution == "lognormal":
        if type(a)==float or type(a)==int: 
            if a<0: a=-np.inf
            else: a=(np.log(a*scale+loc)-loc)/scale
        else: 
            a = np.where(a<0, -np.inf, (np.log(a*scale+loc)-loc)/scale)
        b=(np.log(b*scale+loc)-loc)/scale
        res=np.exp(truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=size))
    elif distribution == "cauchy":
        ua = np.arctan(a) / np.pi + 0.5
        ub = np.arctan(b) / np.pi + 0.5
        U = np.random.uniform(ua, ub, size=size)
        res = loc + scale * np.tan(np.pi * (U - 0.5))
    elif distribution == "translated_weibull" or distribution== "weibull":
        if distribution=="weibull":loc=0
        a=a*scale+loc
        b=b*scale+loc
        a=np.where(a<=loc,loc,a)
        b=np.where(b<=loc,loc,b)

        ua = np.round(weibull_min.cdf(a, c=shape, scale=scale, loc=loc),8)
        ub= np.round(weibull_min.cdf(b, c=shape, scale=scale, loc=loc),8)

        U = np.random.uniform(ua, ub, size=size)
        res = weibull_min.ppf(U, c=shape, scale=scale, loc=loc)
    return res


def truncated_2inter(mean, std, a, b, c, d, shape = 1, distribution="normal",size=1):
    """
    Function to sample from truncated distribution in the union of intervals [a,b] and [c,d].
    """
    if (a >= b) or (c >= d):
        print("a = {}, b = {}, c = {},d = {}".format(a, b, c, d))
    if distribution == "normal":
        F = norm(loc=mean, scale=std).cdf
    elif distribution == "lognormal":
        F = lambda x :  norm.cdf((np.log(x)-mean)/std)
    elif distribution == "cauchy":
        F = cauchy(loc=mean, scale=std).cdf
    elif distribution == "translated_weibull" or distribution== "weibull":
        F= weibull_min(c=shape, scale=std, loc=mean).cdf
        if std<=0: print(mean,std)
        
        
        
    if (b - mean) / std <=  (a - mean) / std : print((b - mean) / std ,  (a - mean) / std)
    if (
        F((b - mean) / std)
        - F((a - mean) / std)
        + F((d - mean) / std)
        - F((c - mean) / std)
    )<=0: return truncated(
            a=(a - mean) / std,
            b=(b - mean) / std,
            size=size,
            loc=mean,
            scale=std,
            distribution=distribution,
        )
    
    elif np.random.uniform(0, 1, 1) < (
        F((b - mean) / std) - F((a - mean) / std)
    ) / (
        F((b - mean) / std)
        - F((a - mean) / std)
        + F((d - mean) / std)
        - F((c - mean) / std)
    ):
        return truncated(
            a=(a - mean) / std,
            b=(b - mean) / std,
            size=size,
            loc=mean,
            scale=std,
            distribution=distribution,
        )
    else:
        return truncated(
            a=(c - mean) / std,
            b=(d - mean) / std,
            size=size,
            loc=mean,
            scale=std,
            distribution=distribution,
        )