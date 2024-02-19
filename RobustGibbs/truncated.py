import numpy as np
from scipy.stats import norm,truncnorm,weibull_min,cauchy,lognorm,genpareto


def truncated(loc, scale, shape=1, a=-np.inf, b=np.inf, distribution="normal", size=1,reparametrization=True):
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
    elif distribution in ["translated_weibull","weibull","generalized_pareto","translated_lognormal"]:
        if distribution=="weibull":loc=0
        a=a*scale+loc
        b=b*scale+loc
        a=np.where(a<=loc,loc,a)
        b=np.where(b<=loc,loc,b)
        if distribution == "weibull" or distribution=="translated_weibull":
            law = weibull_min(c=shape, scale=scale, loc=loc)
        elif distribution=="generalized_pareto":
            if reparametrization: scale,shape = (scale-loc)/2+(scale-loc)**3/(2*shape),1/2-(scale-loc)**2/(2*shape)
            law = genpareto(loc=loc,scale=scale,c=shape)
        elif distribution=="translated_lognormal" or distribution=="lognormal":
            if reparametrization: scale,shape = np.log((scale-loc)**2/np.sqrt((scale-loc)**2+shape)),np.sqrt(np.log(1+shape/(scale-loc)**2))
            law = lognorm(s=shape,scale= np.exp(scale),loc=loc)
        ua,ub =law.cdf(a),law.cdf(b)
        U = np.random.uniform(ua, ub, size=size)
        res = law.ppf(U)
    return res


def truncated_2inter(loc, scale, a, b, c, d, shape = 1, distribution="normal",size=1):
    """
    Function to sample from truncated distribution in the union of intervals [a,b] and [c,d].
    """
    if (a >= b) or (c >= d):
        print("a = {}, b = {}, c = {},d = {}".format(a, b, c, d))
    if distribution == "normal":
        F = norm(loc=loc, scale=scale).cdf
    elif distribution == "lognormal" or distribution=="translated_lognormal":
        F = lambda x :  norm.cdf((np.log(x)-loc)/scale)
    elif distribution == "cauchy":
        F = cauchy(loc=loc, scale=scale).cdf
    elif distribution == "translated_weibull" or distribution== "weibull":
        F= weibull_min(c=shape, scale=scale, loc=loc).cdf
        if scale<=0: print(loc,scale)
    elif distribution=="generalized_pareto":
        F = genpareto(loc=loc,scale=scale,c=shape).cdf
        
        
        
    if (b - loc) / scale <=  (a - loc) / scale : print((b - loc) / scale ,  (a - loc) / scale)
    if (
        F((b - loc) / scale)
        - F((a - loc) / scale)
        + F((d - loc) / scale)
        - F((c - loc) / scale)
    )<=0: return truncated(
            a=(a - loc) / scale,
            b=(b - loc) / scale,
            size=size,
            loc=loc,
            scale=scale,
            distribution=distribution,
        )
    
    elif np.random.uniform(0, 1, 1) < (
        F((b - loc) / scale) - F((a - loc) / scale)
    ) / (
        F((b - loc) / scale)
        - F((a - loc) / scale)
        + F((d - loc) / scale)
        - F((c - loc) / scale)
    ):
        return truncated(
            a=(a - loc) / scale,
            b=(b - loc) / scale,
            size=size,
            loc=loc,
            scale=scale,
            distribution=distribution,
        )
    else:
        return truncated(
            a=(c - loc) / scale,
            b=(d - loc) / scale,
            size=size,
            loc=loc,
            scale=scale,
            distribution=distribution,
        )