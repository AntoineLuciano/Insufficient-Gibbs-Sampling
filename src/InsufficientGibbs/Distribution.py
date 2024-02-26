from typing import Dict, Tuple, List

import numpy as np
from scipy.stats import norm, gamma, lognorm, weibull_min, cauchy, genpareto, pareto, laplace,uniform

# from InsufficientGibbs.Variable import Variable, ContinuousVariable, PositiveContinuousVariable
from Variable import Variable, ContinuousVariable, PositiveContinuousVariable
import matplotlib.pyplot as plt

class Distribution:
    """
    Base class for all distributions
    """
    def __init__(self,
            parameters_dict: Dict[str, Variable],
            name: str) -> None:
        self.name = name
        self.parameters_dict = parameters_dict
    

    def domain(self) -> None:
        """
        Should be overridden by all subclasses
        """
        raise NotImplementedError
    
    def _check_domain(self, X) -> None:
        minn, maxx = self.domain()
        f = lambda x: not(minn < x < maxx)
        if len(list(filter(f, X))) > 0:
            return False 
        return True
        
    def pdf(self, x:List[float]) -> np.ndarray:
        return self._distribution.pdf(x)


    def cdf(self, x:List[float]) -> np.ndarray:
        return self._distribution.cdf(x)


    def logpdf(self, x:List[float]) -> np.ndarray:
        return self._distribution.logpdf(x)

    
    def logcdf(self, x:List[float]) -> np.ndarray:
        return self._distribution.logcdf(x)


    def ppf(self, q:List[float]) -> np.ndarray:
        return self._distribution.ppf(q)
    
    def rvs(self, size:int) -> np.ndarray:
        
        return self._distribution.rvs(size=size)
    
    def truncated(self, a:float, b:float, size:int) -> np.ndarray:
        Fa,Fb =self._distribution.cdf(a),self._distribution.cdf(b)
        U = np.random.uniform(low=Fa, high=Fb, size=size)
        return  self._distribution.ppf(U)
    
    def truncated_2inter(self, a1 : float, b1 : float, a2 : float, b2 : float, size:int) -> np.ndarray:
        Fa1,Fb1 =self._distribution.cdf(a1),self._distribution.cdf(b1)
        Fa2,Fb2 =self._distribution.cdf(a2),self._distribution.cdf(b2)
        p1 = (Fb1-Fa1)/(Fb1-Fa1+Fb2-Fa2)
        U=np.random.uniform(size=size)
        a,b = np.where(U<p1,Fa1,Fa2),np.where(U<p1,Fb1,Fb2)
        V = np.random.uniform(low = a, high = b,size=size)
        return  self._distribution.ppf(V)

    def llikelihood(self, x:List[float]) -> np.ndarray:
        return self._distribution.logpdf(x).sum()



class Normal(Distribution):
    """
    Container for Normal Distribution

    Parameters
    ----------
    loc : float
        location of the normal
    scale : float
        standard deviation of the normal
    name : str


    """
    def __init__(self,
            loc: float=0,
            scale: float=1,
            name: str="",
            theta: list=[]) -> None:
        if theta!=[]:
            loc,scale = theta
        self.loc = ContinuousVariable(loc, name='loc')
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.name = name
        self._distribution = norm(loc=self.loc.value, scale=self.scale.value)
        self.distrib_name = "normal"
        parameters_dict = {'loc': self.loc, 'scale': self.scale}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
class Normal_known_scale(Distribution):
    def __init__(self,
            loc: float=0,
            name: str="",
            theta: list=[]) -> None:
        if theta!=[]:
            loc = theta[0]
        self.loc = ContinuousVariable(loc, name='loc')
        self.name = name
        self._distribution = norm(loc=self.loc.value, scale=1)
        self.distrib_name = "normal"
        parameters_dict = {'loc': self.loc}

        super().__init__(parameters_dict, self.name)
    
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    
class Cauchy(Distribution):
    """
    Container for Cauchy Distribution

    Parameters
    ----------
    loc : float
        location of the Cauchy
    scale : float
        scale of the Cauchy
    name : str


    """
    def __init__(self,
            loc: float=0,
            scale: float=1,
            name: str="",
            theta: list=[]) -> None:
        if theta!=[]:
            loc,scale = theta
        self.loc = ContinuousVariable(loc, name='loc')
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.name = name
        self._distribution = cauchy(loc=self.loc.value, scale=self.scale.value)
        self.distrib_name = "cauchy"
        parameters_dict = {'loc': self.loc, 'scale': self.scale}
        
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))

class Gamma(Distribution):
    """
    Container for Gamma Distribution

    Parameters
    ----------
    scale : float
        scale of the Gamma
    shape : float
        shape of the Gamma
    name : str
    """
    def __init__(self, 
            scale: float=1,
            shape: float=1,
            name: str="",
            theta: list = []) -> None:
        if theta!=[]:
            scale,shape = theta
       
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.shape = PositiveContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = gamma(a=self.shape.value, scale=self.scale.value)
        self.distrib_name = "gamma"
        parameters_dict = {'scale':self.scale, 'shape':self.shape}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
    
class TranslatedGamma(Distribution):
    """
    Container for Translated Gamma Distribution

    Parameters
    ----------
    loc : float
        location of the Gamma
    scale : float
        scale of the Gamma
    shape : float
        shape of the Gamma
    name : str
    """
    def __init__(self, 
            loc: float=0,
            scale: float=1,
            shape: float=1,
            name: str="",
            theta: list = []) -> None:
        if theta!=[]:
            loc,scale,shape = theta
        self.loc = ContinuousVariable(loc, name='loc')
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.shape = PositiveContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = gamma(a=self.shape.value, scale=self.scale.value, loc=self.loc.value)
        self.distrib_name = "gamma"
        parameters_dict = {'loc':self.loc, 'scale':self.scale, 'shape':self.shape}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (self.loc.value, float('inf'))
    
class ReparametrizedGamma(Distribution):
    """
    Container for Reparametrized Gamma Distribution

    Parameters
    ----------
    mean : float
        mean of the Gamma
    std : float
        standard deviation of the Gamma
    name : str
    """
    
    def __init__(self,
                mean: float=1,
                std: float=1,
                name: str="",
                theta: list = []) -> None:
        if theta!=[]:
            mean,std = theta
        self.mean = PositiveContinuousVariable(mean, name='mean')
        self.std = PositiveContinuousVariable(std, name='std')
        self.name = name
        def repar_gamma(mean,std):
            scale,shape = std**2/mean, (mean/std)**2
            return gamma(a=shape, scale=scale)
        
        self._distribution = repar_gamma(mean=self.mean.value,std=self.std.value)
        self.distrib_name = "repar_gamma"
        parameters_dict = {'mean':self.mean, 'std':self.std}
        super().__init__(parameters_dict, self.name)
    
    



class InverseGamma(Distribution):
    """
    Container for Inverse Gamma Distribution
    
    Parameters
    ----------
    scale : float
        scale of the Inverse Gamma
    shape : float
        shape of the Inverse Gamma
    name : str
    """
    
    def __init__(self, 
            scale: float=1,
            shape: float=1,
            name: str="",
            theta: list = []) -> None:
        if theta!=[]:
            scale,shape = theta
       
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.shape = PositiveContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = gamma(a=self.shape.value, scale=self.scale.value)
        self.distrib_name = "inverse_gamma"
        parameters_dict = {'scale':self.scale, 'shape':self.shape}
        
        super().__init__(parameters_dict, self.name)
    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    

class LogNormal(Distribution):
    """
    Container for Lognormal Distribution

    Parameters
    ----------
    scale : float
        log(rv) has loc scale
    shape : float
        log(rv) has scale shape
    name : str
    """
    def __init__(self,
            scale: float=0,
            shape: float=1,
            name: str="",
            theta: list = []) -> None:
        if theta!=[]:
            scale,shape = theta
        self.scale = ContinuousVariable(scale, name='scale')
        self.shape = PositiveContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = lognorm(s=self.shape.value, scale=np.exp(self.scale.value))
        parameters_dict = {'scale':self.scale, 'shape':self.shape}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
class TranslatedLogNormal(Distribution):
    """
    Container for Translated Lognormal Distribution

    Parameters
    ----------
    loc : float
        location of the Lognormal
    scale : float
        log(rv) has loc scale
    shape : float
        log(rv) has scale shape
    name : str
    """
    def __init__(self,
            loc: float=0,
            scale: float=0,
            shape: float=1,
            name: str="",
            theta: list = []) -> None:
        if theta!=[]:
            loc,scale,shape = theta
        self.loc = ContinuousVariable(loc, name='loc')
        self.scale = ContinuousVariable(scale, name='scale')
        self.shape = PositiveContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = lognorm(s=self.shape.value, scale=np.exp(self.scale.value), loc=self.loc.value)
        parameters_dict = {'loc':self.loc, 'scale':self.scale, 'shape':self.shape}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (self.loc.value, float('inf'))
    
class ReparametrizedLogNormal(Distribution):
    """
    Container for Reparametrized Lognormal Distribution

    Parameters
    ----------
    mean : float
        mean of the Lognormal
    std : float
        standard deviation of the Lognormal
    name : str
    """
    
    def __init__(self,
                mean: float=1,
                std: float=1,
                name: str="",
                theta: list = []) -> None:
        if theta!=[]:
            mean,std = theta
        self.mean = PositiveContinuousVariable(mean, name='mean')
        self.std = PositiveContinuousVariable(std, name='std')
        self.name = name
        def repar_lognorm(mean,std):
            scale,shape = np.log((mean**2)/np.sqrt(mean**2+std**2)),np.sqrt(np.log(1+std**2/(mean**2)))
            return lognorm(s=shape, scale=np.exp(scale))
        
        self._distribution = repar_lognorm(mean=self.mean.value,std=self.std.value)
        parameters_dict = {'mean':self.mean, 'std':self.std}
        super().__init__(parameters_dict, self.name)
    
    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
        
class Weibull(Distribution):
    """
    Container for Weibull Distribution

    Parameters
    ----------
    scale : float
        scale of the weibull - equivalent to 1/rate
    shape : float
        shape of the weibull
    name : str
    """
    def __init__(self, 
            scale: float=1,
            shape: float=1,
            name: str="",
            theta: list=[]) -> None:
        if theta!=[]:
            scale,shape = theta
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.shape = PositiveContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = weibull_min(c=self.shape.value, scale=self.scale.value)
        parameters_dict = {'scale':self.scale, 'shape':self.shape}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
class TranslatedWeibull(Distribution):
    """
    Container for Translated Weibull Distribution

    Parameters
    ----------
    loc : float
        location of the weibull
    scale : float
        scale of the weibull - equivalent to 1/rate
    shape : float
        shape of the weibull
    name : str
    """
    def __init__(self, 
            loc: float=0,
            scale: float=1,
            shape: float=1,
            name: str="",
            theta: list = []) -> None:
        if theta!=[]:
            loc,scale,shape = theta
        self.loc = ContinuousVariable(loc, name='loc')
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.shape = PositiveContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = weibull_min(loc = self.loc.value, c=self.shape.value, scale=self.scale.value)
        parameters_dict = {"loc" : self.loc, 'scale':self.scale, 'shape':self.shape}
        
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (self.loc.value, float('inf'))
    

    
class Laplace(Distribution):
    def __init__(self,
                 loc: float=0,
                 scale: float=1,
                 name: str="",
                 theta: list = []) -> None:
        if theta!=[]:
            loc,scale = theta
        self.loc = ContinuousVariable(loc, name='loc')
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.name = name
        self._distribution = laplace(loc=loc,scale=scale)
        parameters_dict = {'loc': self.loc, 'scale':self.scale}
        super().__init__(parameters_dict, self.name)
        
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))

class Laplace_known_scale(Distribution):
    def __init__(self,
                 loc: float=0,
                 name: str="",
                 theta: list = []) -> None:
        if theta!=[]:
            loc = theta[0]
        self.loc = ContinuousVariable(loc, name='loc')
        self.name = name
        self._distribution = laplace(loc=loc,scale=1/np.sqrt(2))
        parameters_dict = {'loc': self.loc}
        super().__init__(parameters_dict, self.name)
        
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
class GeneralizedPareto(Distribution):
    """
    Container for Generalized Pareto Distribution

    Parameters
    ----------
    loc : float
        location of the generalized pareto
    scale : float
        scale of the generalized pareto
    shape : float
        shape of the generalized pareto
    name : str
    """
    def __init__(self, 
            loc: float=0,
            scale: float=1,
            shape: float=1,
            name: str="",
            theta: list = []) -> None:
        if theta!=[]:
            loc,scale,shape = theta
        self.loc = ContinuousVariable(loc, name='loc')
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.shape = ContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = genpareto(c=self.shape.value, scale=self.scale.value, loc=self.loc.value)
        parameters_dict = {'loc':self.loc, 'scale':self.scale, 'shape':self.shape}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
class Pareto(Distribution):
    """
    Container for Pareto Distribution

    Parameters
    ----------
    scale : float
        scale of the pareto
    shape : float
        shape of the pareto
    name : str
    """
    def __init__(self, 
            scale: float=1,
            shape: float=1,
            name: str="") -> None:
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.shape = PositiveContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = pareto(b=self.shape.value, scale=self.scale.value)
        parameters_dict = {'scale':self.scale, 'shape':self.shape}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["scale"], float('inf'))
    
    
class pareto2:
    def __init__(self, loc=0, scale=1,shape=1):
        self.loc = loc
        self.scale = scale
        self.shape = shape
        
    def pdf(self,x):
        x=np.array(x)
        return np.where(x>=self.loc,(self.shape/self.scale)*(1+(x-self.loc)/self.scale)**(-self.shape-1),0)
    
    def cdf(self,x):
        x=np.array(x)
        return np.where(x>=self.loc,1-(1+(x-self.loc)/self.scale)**(-self.shape),0)
    def ppf(self,x):
        x=np.array(x)
        return self.loc+self.scale*((1-x)**(-1/self.shape)-1)
    def logpdf(self,x):
        x=np.array(x)
        return np.where(x>=self.loc,np.log(self.shape)-np.log(self.scale)-(self.shape+1)*np.log(1+(x-self.loc)/self.scale),-np.inf)
    def rvs(self,size):
        return self.ppf(np.random.random(size))
    def mean(self):
        if self.shape<1:
            return np.inf
        else:
            return self.scale/(self.shape-1)+self.loc
    def std(self):
        if self.shape<2:
            return np.inf
        else:
            return self.scale*np.sqrt(self.shape/((self.shape-1)**2*(self.shape-2)))
        
class ParetoType2(Distribution):
    
    def __init__(self,
            loc: float=0,
            scale: float=1,
            shape: float=1,
            name: str="",
            theta: list = []) -> None:
        if theta!=[]:
            loc,scale,shape = theta
        self.loc = ContinuousVariable(loc, name='loc')
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.shape = PositiveContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = pareto2(loc=self.loc.value, scale=self.scale.value,shape=self.shape.value)
        parameters_dict = {'loc': self.loc, 'scale':self.scale, 'shape':self.shape}
        super().__init__(parameters_dict, self.name)
        
        
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
    
    